"""Discover and stage MapD5 program releases. Installation runs after the GUI exits."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid
from zipfile import BadZipFile,ZipFile

from .s1_updates import REPO
from .s1_update_worker import INSTALL_SCRIPT

APP_TAG='v3.7.26-rc.1'
API_ROOT=f'https://api.github.com/repos/{REPO}'
PROGRAM='MapD5.exe'
FULL_NAME='MapD5-Windows-x64.zip'
FULL_FILES={PROGRAM,'maps.zip','README.md','LICENSE','SOURCES.md','更新说明.txt'}
FULL_DIRS={'_internal','notices'}
MAX_ARCHIVE=512*1024**2
TX_PREFIX='.mapd5-update-'
TOKEN=re.compile(r'[0-9a-f]{32}\Z')
TAG=re.compile(r'v(0|[1-9]\d{0,3})\.(0|[1-9]\d{0,3})\.(0|[1-9]\d{0,3})(?:-(alpha|beta|rc)\.(0|[1-9]\d{0,4}))?\Z')


def version(tag):
    m=TAG.fullmatch(tag)
    if not m:return None
    a,b,c,stage,number=m.groups()
    return int(a),int(b),int(c),{'alpha':0,'beta':1,'rc':2,None:3}[stage],int(number or 0)


@dataclass(frozen=True)
class Release:
    tag: str
    title: str
    asset_id: int
    name: str
    size: int
    digest: str
    url: str

    @property
    def full(self):
        return self.name==FULL_NAME


def choose_release(rows,current=APP_TAG):
    installed=version(current)
    releases=[]
    for row in rows:
        tag=row.get('tag_name','')
        key=version(tag)
        if row.get('draft') or not key or key<=installed:continue
        if installed[3]==3 and (row.get('prerelease') or key[3]!=3):continue
        releases.append((key,row))
    if not releases:return None
    key,row=max(releases,key=lambda pair:pair[0])
    tag=row['tag_name']
    core='.'.join(str(n) for n in key[:3])
    names=[FULL_NAME]
    if key[:2]==installed[:2]:names.insert(0,f'MapD5-{key[0]}.{key[1]}.x-to-{core}.zip')
    for name in names:
        for asset in row.get('assets',[]):
            if asset.get('name')!=name or asset.get('state')!='uploaded':continue
            url=f'https://github.com/{REPO}/releases/download/{tag}/{name}'
            digest=asset.get('digest') or ''
            if (asset.get('browser_download_url')!=url or not re.fullmatch(r'sha256:[0-9a-f]{64}',digest)
                    or type(asset.get('id')) is not int or asset['id']<1 or type(asset.get('size')) is not int
                    or not 0<asset['size']<=MAX_ARCHIVE):
                continue
            return Release(tag,row.get('name') or tag,asset['id'],name,asset['size'],digest[7:],url)
    raise ValueError(f'{tag} 已发布，暂时没有适用于此安装且可校验的 Windows 更新包。')


def _open(url,accept):
    req=Request(url,headers={'User-Agent':'MapD5-program-updater','Accept':accept,'X-GitHub-Api-Version':'2022-11-28'})
    resp=urlopen(req,timeout=18)
    dest=urlsplit(resp.geturl());host=dest.hostname or ''
    if dest.scheme!='https' or not (host in ('github.com','api.github.com') or host.endswith('.githubusercontent.com')):
        resp.close();raise ValueError('更新请求转向了未知的下载站点。')
    return resp


def latest_release(current=APP_TAG):
    rows=[]
    for page in range(1,21):
        with _open(f'{API_ROOT}/releases?per_page=100&page={page}','application/vnd.github+json') as response:
            data=response.read(8*1024**2+1)
        if len(data)>8*1024**2:raise ValueError('版本列表超过大小限制。')
        batch=json.loads(data)
        if not isinstance(batch,list):raise ValueError('GitHub 返回的版本列表无效。')
        rows.extend(batch)
        if len(batch)<100:return choose_release(rows,current)
    raise ValueError('未能完整读取 GitHub 发布记录。')


def download(release,target,progress):
    failures=[]
    for url in (f'{API_ROOT}/releases/assets/{release.asset_id}',release.url):
        try:
            with _open(url,'application/octet-stream') as source,target.open('wb') as out:
                total=0;digest=hashlib.sha256()
                while chunk:=source.read(256*1024):
                    total+=len(chunk)
                    if total>release.size:raise ValueError('下载大小与发布记录不符。')
                    digest.update(chunk);out.write(chunk);progress(total,release.size)
            if total!=release.size or digest.hexdigest()!=release.digest:
                raise ValueError('下载文件与 GitHub 的 SHA-256 不符。')
            return
        except (HTTPException,URLError,OSError,ValueError) as exc:
            failures.append(str(exc))
    raise ValueError('无法取得校验正确的更新包。'+failures[-1])


def unpack(archive,destination,full):
    seen=set();total=0
    with ZipFile(archive) as z:
        for info in z.infolist():
            name=info.filename.rstrip('/')
            parts=name.split('/')
            if (not name or '\\' in name or any(not part or part in ('.','..') or part.endswith((' ','.'))
                    or re.search(r'[<>:"|?*\x00-\x1f]',part) or PureWindowsPath(part).is_reserved() for part in parts)
                    or stat.S_ISLNK(info.external_attr>>16)):
                raise ValueError('压缩包包含无效路径。')
            if full:
                if parts[0]!='MapD5':raise ValueError('完整包目录不匹配。')
                parts=parts[1:]
                if not parts and info.is_dir():continue
                if not parts or (parts[0] not in FULL_DIRS and not (len(parts)==1 and parts[0] in FULL_FILES)):
                    raise ValueError('完整包包含无法更新的组件。')
            elif parts==['更新说明.txt']:continue
            elif parts!=[PROGRAM] or info.is_dir():raise ValueError('补丁结构无效。')
            key='/'.join(parts).casefold()
            if key in seen:raise ValueError('压缩包包含重名文件。')
            seen.add(key);total+=info.file_size
            if total>2*1024**3 or len(seen)>20000:raise ValueError('解压文件超过大小限制。')
            path=destination.joinpath(*parts)
            if info.is_dir():path.mkdir(parents=True,exist_ok=True);continue
            path.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,path.open('xb') as out:shutil.copyfileobj(src,out,256*1024)
    exe=destination/PROGRAM
    with exe.open('rb') as fh:
        if fh.read(2)!=b'MZ':raise ValueError('更新包没有有效的 Windows 程序。')
    if full and not (destination/'_internal').is_dir():raise ValueError('完整包缺少运行环境。')


def file_hash(path):
    with path.open('rb') as fh:return hashlib.file_digest(fh,'sha256').hexdigest()


def prepare(release,exe,progress):
    if exe.name.casefold()!=PROGRAM.casefold():raise ValueError('请从解压后的 MapD5.exe 使用程序更新。')
    token=uuid.uuid4().hex
    tx=exe.parent/(TX_PREFIX+token)
    tx.mkdir()
    try:
        download(release,tx/'download.zip',progress)
        new=tx/'new';new.mkdir()
        unpack(tx/'download.zip',new,release.full)
        (tx/'download.zip').unlink()
        (tx/'tag').write_text(release.tag,'ascii')
        return tx
    except BaseException:
        shutil.rmtree(tx,ignore_errors=True)
        raise


def launch_worker(exe,tx,resume):
    # All paths are data in the environment, never interpolated into shell code.
    powershell=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    env=dict(os.environ,MAPD5_INSTALL=str(exe.parent),MAPD5_TX=tx.name,MAPD5_PARENT=str(os.getpid()),
             MAPD5_OLD_HASH=file_hash(exe),MAPD5_NEW_HASH=file_hash(tx/'new'/PROGRAM),
             MAPD5_RESUME='1' if resume else '0',PYINSTALLER_RESET_ENVIRONMENT='1')
    env.pop('PSMODULEPATH',None)  # Let Windows PowerShell load its own system modules.
    worker=subprocess.Popen([str(powershell),'-NoProfile','-NonInteractive','-Command',INSTALL_SCRIPT],
                            cwd=str(powershell.parent),env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    for _ in range(250):
        if (tx/'prepared').is_file():return
        if worker.poll() is not None:break
        time.sleep(.1)
    if worker.poll() is None:
        worker.kill();worker.wait(timeout=10)
    error=tx/'error'
    raise OSError(error.read_text('utf-8-sig') if error.is_file() else '更新交接没有启动，当前程序仍可使用。')


def startup_receipt(exe,token,ready=False):
    if not TOKEN.fullmatch(token):raise ValueError('更新启动标记无效。')
    tx=exe.parent/(TX_PREFIX+token)
    if (tx/'tag').read_text('ascii')!=APP_TAG:raise ValueError('下载的程序版本与发布标签不符。')
    if ready:
        pending=tx/'started.tmp'
        pending.write_text(APP_TAG,'ascii')
        os.replace(pending,tx/'started')


def readable_error(exc):
    if isinstance(exc,PermissionError):return '安装目录不可写或文件正被其他程序使用。请从可写目录启动后重试。'
    if isinstance(exc,BadZipFile):return '更新包无法解压，请重新检查并下载。'
    if isinstance(exc,HTTPError) and exc.code in (403,429):return 'GitHub 限制了版本查询，请稍后重试。'
    if isinstance(exc,(URLError,TimeoutError)):return '无法连接 GitHub，请检查网络后重试。'
    return str(exc)
