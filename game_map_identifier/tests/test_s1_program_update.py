from io import BytesIO
from pathlib import Path
import hashlib
import sys
from unittest.mock import patch
from zipfile import ZipFile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import s1_program_update as u


def release(tag,file=None):
    name=file or 'MapD5-3.7.x-to-'+tag[1:].split('-')[0]+'.zip'
    return {'tag_name':tag,'name':tag,'draft':False,'prerelease':'-' in tag,'assets':[
        {'id':9,'name':name,'state':'uploaded','size':17,'digest':'sha256:'+'a'*64,
         'browser_download_url':f'https://github.com/{u.REPO}/releases/download/{tag}/{name}'}]}


def test_sort_by_version_not_list_order():
    rows=[release('v3.7.9'),release('v3.7.10-rc.2'),release('v3.7.10-rc.12')]
    assert u.choose_release(rows,'v3.7.10-rc.1').tag=='v3.7.10-rc.12'
    assert u.choose_release(rows,'v3.7.9') is None
    rows.append(release('v3.7.10'))
    assert u.choose_release(rows,'v3.7.10-rc.12').tag=='v3.7.10'
    assert u.choose_release(rows,'v3.7.10') is None


def test_cross_series_uses_full_and_draft_is_ignored():
    official=release('v4.0.0',u.FULL_NAME)
    rows=[dict(release('v5.0.0'),draft=True),official]
    assert u.choose_release(rows,'v3.7.16-rc.1').full
    with pytest.raises(ValueError):u.choose_release([release('v4.0.0')],'v3.7.16-rc.1')


@pytest.mark.parametrize('field,value',[('browser_download_url','https://other.invalid/file'),('size',0),('digest',None)])
def test_unverified_asset_is_not_selected(field,value):
    row=release('v3.7.17')
    row['assets'][0][field]=value
    with pytest.raises(ValueError):u.choose_release([row],'v3.7.16')


@pytest.mark.parametrize('path',['../escape','MapD5/../escape','MapD5/C:/evil','MapD5/_internal/CON','MapD5/_internal/a.'])
def test_invalid_archive_paths_never_leave_stage(tmp_path,path):
    z=tmp_path/'bad.zip'
    with ZipFile(z,'w') as f:f.writestr(path,b'bad')
    with pytest.raises(ValueError):u.unpack(z,tmp_path/'new',True)
    assert not (tmp_path/'escape').exists()


def test_corruption_or_disconnect_preserves_installed_executable(tmp_path):
    exe=tmp_path/u.PROGRAM;exe.write_bytes(b'old')
    rel=u.choose_release([release('v3.7.17')],'v3.7.16')
    with patch.object(u,'_open',side_effect=lambda *args:BytesIO(b'wrong')):
        with pytest.raises(ValueError):u.prepare(rel,exe,lambda *_:None)
    assert exe.read_bytes()==b'old' and list(tmp_path.iterdir())==[exe]
    with patch.object(u,'_open',side_effect=OSError('offline')):
        with pytest.raises(ValueError):u.prepare(rel,exe,lambda *_:None)
    assert exe.read_bytes()==b'old' and list(tmp_path.iterdir())==[exe]


def test_verified_archive_stages_but_does_not_overwrite(tmp_path):
    b=BytesIO()
    with ZipFile(b,'w') as z:z.writestr('MapD5.exe',b'MZ binary')
    payload=b.getvalue();row=release('v3.7.17')
    row['assets'][0].update(size=len(payload),digest='sha256:'+hashlib.sha256(payload).hexdigest())
    exe=tmp_path/u.PROGRAM;exe.write_bytes(b'previous')
    with patch.object(u,'_open',side_effect=lambda *args:BytesIO(payload)):
        tx=u.prepare(u.choose_release([row],'v3.7.16'),exe,lambda *_:None)
    assert (tx/'new'/u.PROGRAM).read_bytes()==b'MZ binary'
    assert (tx/'tag').read_text()=='v3.7.17'
    assert exe.read_bytes()==b'previous'


def test_acknowledgement_requires_exact_version_and_transaction(tmp_path):
    exe=tmp_path/u.PROGRAM;token='a'*32
    tx=tmp_path/(u.TX_PREFIX+token);tx.mkdir()
    (tx/'tag').write_text('v3.7.18')
    with pytest.raises(ValueError):u.startup_receipt(exe,token,ready=True)
    assert not (tx/'started').exists()
    with pytest.raises(ValueError):u.startup_receipt(exe,'../../other',ready=True)
    (tx/'tag').write_text(u.APP_TAG);u.startup_receipt(exe,token,ready=True)
    assert (tx/'started').read_text()==u.APP_TAG
