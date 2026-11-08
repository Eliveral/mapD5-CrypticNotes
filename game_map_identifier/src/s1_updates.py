"""Verified data-only S1 map installation in the user's profile."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from threading import Lock
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image
from .s1_catalog import S1Catalog

REPO='Eliveral/mapD5-CrypticNotes'
FEED=f'https://raw.githubusercontent.com/{REPO}/main/updates/stable.json'
API=1
PREFIX='assets/s1_phosphor/'
CONFIG={'config/s1_landmarks.json','config/s1_hd_registration.json','config/s1_floor_relations.json'}
TABLES={PREFIX+'catalog.json',PREFIX+'floors.json',PREFIX+'ui/index.json'} | CONFIG
RASTER=re.compile(r'assets/s1_phosphor/(hd|structure|ui)/[A-Za-z0-9_]+\.png\Z')
DIGEST=re.compile(r'[0-9a-f]{64}\Z')


def profile():
    return Path(os.environ.get('LOCALAPPDATA',Path.home()))/'CrypticNotesAssistantS1'


def read_json(path):
    return path.read_text('utf-8-sig')


def active(bank):
    try:
        name=json.loads(read_json(bank/'current.json'))['folder']
        if not re.fullmatch(r'[1-9][0-9]{0,11}-[0-9a-f]{16}',name):return None
        root=bank/name
        if (root/PREFIX/'catalog.json').is_file():return root
    except (OSError,ValueError,KeyError,TypeError):pass
    return None


def map_root(application):
    installed=active(profile()/'maps')
    return (installed or application)/PREFIX


def record(path):
    return json.loads(read_json(path/'map-pack.json')) if (path/'map-pack.json').is_file() else {}


def request(url):
    return urlopen(Request(url,headers={'User-Agent':'mapD5-CrypticNotes/3.7'}),timeout=25)


def fetch(url,target,limit,expected):
    if not DIGEST.fullmatch(expected):raise ValueError('文件校验值无效')
    digest=hashlib.sha256();length=0
    with request(url) as response,target.open('wb') as out:
        while chunk:=response.read(131072):
            length+=len(chunk)
            if length>limit:raise ValueError('下载文件超过大小限制')
            out.write(chunk);digest.update(chunk)
    if digest.hexdigest()!=expected:raise ValueError('下载内容校验失败，请稍后重试')


def check_release():
    with request(FEED) as response:data=json.loads(response.read(65537))
    if data['format']!=API or type(data['version'])!=int or data['version']<=0:
        raise ValueError('这份地图需要更新版本的助手')
    if not data['url'].startswith(f'https://github.com/{REPO}/releases/download/') or not DIGEST.fullmatch(data['sha256']):
        raise ValueError('更新目录中的地图下载信息无效')
    return data


@contextmanager
def install_lock(bank):
    import msvcrt
    bank.mkdir(parents=True,exist_ok=True)
    with (bank/'install.lock').open('a+b') as handle:
        if not handle.tell():handle.write(b'0');handle.flush()
        handle.seek(0)
        try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:raise ValueError('另一个窗口正在安装地图') from exc
        try:yield
        finally:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


def validate(root):
    maps=root/PREFIX
    cat=S1Catalog(maps)
    errors=cat.validate()
    if errors:raise ValueError(errors[0])
    floors=json.loads(read_json(maps/'floors.json'))['floors']
    matrices=json.loads(read_json(root/'config/s1_hd_registration.json'))
    points=json.loads(read_json(root/'config/s1_landmarks.json'))
    ids={f['id'] for f in floors}
    if set(matrices)!=ids or not set(points)<=ids:raise ValueError('地图坐标索引与楼层不一致')
    for f in floors:
        level=f['floor'];key=f['id'];arr=np.array(matrices[key],dtype=float)
        if arr.shape!=(2,3) or not np.isfinite(arr).all() or abs(np.linalg.det(arr[:,:2]))<1e-6:
            raise ValueError('楼层配准数据无效')
        width,height=cat.by_id['hd_'+key.removeprefix('s1_')]['size']
        markers=points.get(key,{})
        required = {'B1':set(), 'B2':{'supply'}, 'B3':{'supply','star'}}[level]
        optional = {'B1':set(), 'B2':{'race'}, 'B3':{'alchemy'}}
        allowed = required | optional[level]
        if not required <= set(markers) <= allowed:
            raise ValueError('房间标注不完整')
        for x,y in markers.values():
            if not (0<=x<width and 0<=y<height):raise ValueError('房间标注超出地图')
    from .s1_floor_transition import FloorCoordinates
    from .s1_visibility import MapVisibility
    FloorCoordinates(cat)
    ui=json.loads(read_json(maps/'ui/index.json'))['templates']
    if {row['id'] for row in ui}!={'close','zoom_plus'}:raise ValueError('地图界面模板不完整')
    for row in ui:
        name=row['file']
        if not re.fullmatch(r'[a-z_]+\.png',name) or hashlib.sha256((maps/'ui'/name).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('地图界面模板校验失败')
    MapVisibility(maps/'ui')


def install(archive,bank,progress=lambda text:None,expected_version=None):
    with install_lock(bank), tempfile.TemporaryDirectory(prefix='pending-',dir=bank) as tmp:
        root=Path(tmp)/'maps';root.mkdir()
        with ZipFile(archive) as z:
            members=z.infolist()
            if len(members)>3000 or len({m.filename.lower() for m in members})!=len(members) or sum(m.file_size for m in members)>512*1024**2:
                raise ValueError('地图包大小或目录无效')
            for m in members:
                if m.filename not in TABLES|{'map-pack.json'} and not RASTER.fullmatch(m.filename):
                    raise ValueError('地图包包含不支持的文件')
                if m.file_size>32*1024**2:raise ValueError('地图文件过大')
            meta=json.loads(z.read('map-pack.json'))
            version=meta['version']
            if meta['format']!=API or type(version)!=int or not 0<version<=999999999999 or (expected_version is not None and version!=expected_version):
                raise ValueError('地图版本不兼容')
            current=active(bank)
            if current and version<record(current)['version']:raise ValueError('已安装更高版本的地图')
            files=meta['files'];remote=meta.get('downloads',{})
            if not TABLES<=files.keys() or set(files)&set(remote) or set(z.namelist())!={'map-pack.json',*files}:
                raise ValueError('地图包内容与清单不一致')
            if len(remote)>600:raise ValueError('地图图层过多')
            for name,sha in files.items():
                data=z.read(name)
                if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('地图包校验失败：'+name)
                dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        # Only map pixels may be retrieved externally. Each URL pins a Git blob's
        # revision and has a separate byte digest checked before activation.
        remaining=512*1024**2-sum(m.file_size for m in members)
        budget_lock=Lock()
        def download(item):
            nonlocal remaining
            name,info=item
            if not RASTER.fullmatch(name) or not name.startswith(PREFIX+'hd/'):raise ValueError('下载目标无效')
            parts=urlsplit(info['url'])
            if (parts.scheme!='https' or parts.netloc!='raw.githubusercontent.com' or
                not re.fullmatch(r'/[\w.-]+/[\w.-]+/[0-9a-f]{40}/[^?#]+\.png',parts.path)):
                raise ValueError('底图来源无效')
            dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
            fetch(info['url'],dest,8*1024**2,info['sha256'])
            with budget_lock:
                remaining-=dest.stat().st_size
                if remaining<0:raise ValueError('地图包总大小超出限制')
        with ThreadPoolExecutor(max_workers=5) as pool:
            jobs=[pool.submit(download,row) for row in remote.items()]
            for i,job in enumerate(as_completed(jobs),1):
                job.result();progress(f'高清底图 {i} / {len(jobs)}')
        for name in (*files,*remote):
            if name.endswith('.png'):
                with Image.open(root/name) as im:
                    if im.format!='PNG' or max(im.size)>8192 or im.width*im.height>20000000:
                        raise ValueError('地图像素尺寸无效')
        progress('正在核对地图、楼层和房名…')
        validate(root)
        (root/'map-pack.json').write_text(json.dumps(meta,ensure_ascii=False),'utf-8')
        key=f'{version}-'+hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()[:16]
        dest=bank/key
        if dest.exists():validate(dest)
        else:root.rename(dest)
        pointer=Path(tmp)/'current.json';pointer.write_text(json.dumps({'folder':key}),'utf-8');pointer.replace(bank/'current.json')
        return dest
