"""Bounded local evidence for actual floor transitions; never sends input."""
import json
import hashlib
import threading
import time
import cv2

class TransitionRecorder:
    def __init__(self,root):
        self.root=root/'transitions'
        self.busy=False
        self.key=None
        self.count=0
        self.saved_at=0.
        self.slot=0

    @staticmethod
    def picture(frame):
        # Recognition evidence must preserve the pixels actually evaluated.
        return frame.copy(),1.

    def observe(self,sample,source):
        sample=dict(sample);source=dict(source)
        sample['shared_attempt']=dict(sample.get('shared_attempt') or {})
        key=(sample.get('revision'),source['floor']['id'],sample.get('selected_floor'),source.get('sample_at',source['at']))
        now=time.monotonic()
        if self.busy:return
        if key!=self.key:self.key=key;self.count=0
        if self.count>=2 or (self.count and now-self.saved_at<.65):return
        sample['frame']=sample['frame'].copy()
        if sample.get('clear_mask') is not None:sample['clear_mask']=sample['clear_mask'].copy()
        source['matrix']=source['matrix'].copy()
        if source.get('player_hd') is not None:source['player_hd']=source['player_hd'].copy()
        self.busy=True;self.count+=1;self.saved_at=now
        slot=self.slot;self.slot=(slot+1)%24
        def save():
            try:
                self.root.mkdir(parents=True,exist_ok=True)
                frame,scale=self.picture(sample['frame'])
                name=f'transition-{slot}'
                files={}
                def picture_file(label,pixels):
                    path=self.root/(name+'-'+label+'.png')
                    data=cv2.imencode('.png',pixels)[1].tobytes()
                    tmp=path.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(path)
                    files[path.name]=hashlib.sha256(data).hexdigest()
                picture_file('target',frame)
                original=source.get('diagnostic_frame')
                if original is not None:picture_file('source',original)
                clear=sample.get('clear_mask')
                if clear is not None:
                    clear=cv2.resize(clear,frame.shape[1::-1],interpolation=cv2.INTER_NEAREST)
                    picture_file('clear',clear)
                point=source.get('player_hd')
                payload=dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),source=source['floor']['id'],
                    target=sample.get('selected_floor'),source_matrix=source['matrix'].tolist(),
                    source_zoom=source.get('zoom'),target_zoom=sample.get('zoom'),
                    player_hd=point.tolist() if point is not None else None,
                    source_scale=source.get('diagnostic_scale'),target_scale=scale,
                    has_source=original is not None,has_clear=clear is not None,files=files,
                    attempt=sample.get('shared_attempt'),source_at=source.get('sample_at'),target_at=sample['at'],
                    scope='Native app capture; projection accuracy and action classification require separate verification')
                tmp=self.root/(name+'.tmp')
                tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),'utf8')
                tmp.replace(self.root/(name+'.json'))
            except (OSError,ValueError,cv2.error):pass
            finally:self.busy=False
        threading.Thread(target=save,daemon=True).start()
