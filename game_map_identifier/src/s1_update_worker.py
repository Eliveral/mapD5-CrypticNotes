"""Windows releases the running executable only after its owning process exits."""
INSTALL_SCRIPT=r"""
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath($env:MAPD5_INSTALL)
if ($env:MAPD5_TX -cnotmatch '^\.mapd5-update-([0-9a-f]{32})$') { exit 2 }
$token=$Matches[1]
$tx=[IO.Path]::GetFullPath((Join-Path $root $env:MAPD5_TX))
if ([IO.Path]::GetDirectoryName($tx) -ne $root) { exit 2 }
$exe=Join-Path $root 'MapD5.exe'
$new=Join-Path $tx 'new'
$backup=Join-Path $tx 'backup'
$utf8=New-Object Text.UTF8Encoding($false)
function WriteText($path,$value) { [IO.File]::WriteAllText($path,$value,$utf8) }
function FileHash($path) {
    $stream=[IO.File]::OpenRead($path)
    $sha=[Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','') }
    finally { $sha.Dispose();$stream.Dispose() }
}
function StartAssistant([string]$arguments) {
    $info=New-Object Diagnostics.ProcessStartInfo
    $info.FileName=$exe
    $info.WorkingDirectory=$root
    $info.Arguments=$arguments
    $info.UseShellExecute=$false
    return [Diagnostics.Process]::Start($info)
}
function MoveEntry($a,$b) {
    if ([IO.Directory]::Exists($a)) { [IO.Directory]::Move($a,$b) }
    else { [IO.File]::Move($a,$b) }
}
function CleanTree($folder) {
    $dir=New-Object IO.DirectoryInfo($folder)
    if (($dir.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
        foreach ($entry in $dir.GetFileSystemInfos()) {
            if ($entry -is [IO.DirectoryInfo]) { CleanTree $entry.FullName }
            else { [IO.File]::Delete($entry.FullName) }
        }
    }
    [IO.Directory]::Delete($folder,$false)
}
$allowed=@('_internal','notices','maps.zip','README.md','LICENSE','SOURCES.md','更新说明.txt','MapD5.exe')
$changes=New-Object 'Collections.Generic.List[string]'
$existed=@{}
$closed=$false
$child=$null
$lock=$null
$finished=$false
$restartArgs=@()
if ($env:MAPD5_RESUME -ne '1') { $restartArgs+= '--standby' }
try {
    $lock=[IO.File]::Open((Join-Path $root '.mapd5-update.lock'),'OpenOrCreate','ReadWrite','None')
    $parent=[Diagnostics.Process]::GetProcessById([int]$env:MAPD5_PARENT)
    if ($parent.MainModule.FileName -ne $exe) { throw '更新来源与这份安装不一致。' }
    $tag=[IO.File]::ReadAllText((Join-Path $tx 'tag'))
    $names=@(Get-ChildItem -LiteralPath $new -Force | ForEach-Object { $_.Name })
    foreach ($name in $names) { if ($allowed -cnotcontains $name) { throw '不支持的更新组件。' } }
    if ($names -cnotcontains 'MapD5.exe') { throw '缺少主程序。' }
    if ((FileHash $exe) -ne $env:MAPD5_OLD_HASH -or
        (FileHash (Join-Path $new 'MapD5.exe')) -ne $env:MAPD5_NEW_HASH) { throw '程序文件已变化，请重新检查更新。' }
    [void][IO.Directory]::CreateDirectory($backup)
    WriteText (Join-Path $tx 'prepared') 'ok'
    if (-not $parent.WaitForExit(60000)) { throw '旧助手仍在运行。请关闭后重试。' }
    $closed=$true
    foreach ($name in (@($names | Where-Object {$_ -cne 'MapD5.exe'})+@('MapD5.exe'))) {
        $dest=Join-Path $root $name
        $source=Join-Path $new $name
        $old=Join-Path $backup $name
        $existed[$name]=[IO.File]::Exists($dest) -or [IO.Directory]::Exists($dest)
        [void]$changes.Add($name)
        if ([IO.File]::Exists($dest) -and [IO.File]::Exists($source)) {
            [IO.File]::Replace($source,$dest,$old)
        } else {
            if ($existed[$name]) { MoveEntry $dest $old }
            MoveEntry $source $dest
        }
    }
    $child=StartAssistant ((@('--program-update-ready',$token)+$restartArgs) -join ' ')
    $started=Join-Path $tx 'started'
    $timer=[Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt 75 -and -not $child.HasExited) {
        if ([IO.File]::Exists($started)) {
            try { if ([IO.File]::ReadAllText($started) -ceq $tag) { $finished=$true;break } }
            catch [IO.IOException] { }
        }
        Start-Sleep -Milliseconds 150
    }
    if (-not $finished) { throw '新版本未完成启动。' }
    WriteText (Join-Path $root '.mapd5-last-update.txt') ('已成功更新到 '+$tag)
} catch {
    $finished=$false
    $reason=$_.Exception.Message
    if ($closed) {
        try {
            if ($null -ne $child -and -not $child.HasExited) { $child.Kill();$child.WaitForExit() }
            $broken=Join-Path $tx 'failed'
            [void][IO.Directory]::CreateDirectory($broken)
            for ($i=$changes.Count-1;$i -ge 0;$i--) {
                $name=$changes[$i];$dest=Join-Path $root $name;$old=Join-Path $backup $name
                $hasBackup=[IO.Directory]::Exists($old) -or [IO.File]::Exists($old)
                if ($hasBackup -or -not $existed[$name]) {
                    if (Test-Path -LiteralPath $dest) { MoveEntry $dest (Join-Path $broken $name) }
                    if ($hasBackup) { MoveEntry $old $dest }
                }
            }
            $reason='已恢复原版。'+$reason
            WriteText (Join-Path $root '.mapd5-last-update.txt') $reason
            [void](StartAssistant ($restartArgs -join ' '))
            $finished=$true
        } catch {
            $reason+=' 恢复未完成，备份位于 '+$backup+'。'+$_.Exception.Message
            WriteText (Join-Path $root '.mapd5-last-update.txt') $reason
        }
    }
    WriteText (Join-Path $tx 'error') $reason
} finally {
    if ($null -ne $lock) { $lock.Dispose() }
    if ($finished) { try { CleanTree $tx } catch {} }
}
"""
