"""Bounded local evidence for actual floor transitions; never sends input."""
import json
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
        scale=min(1.,1280/frame.shape[1])
        return cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA),scale

    def observe(self,sample,source):
        key=(sample.get('revision'),source['floor']['id'],sample.get('selected_floor'),source.get('sample_at',source['at']))
        now=time.monotonic()
        if self.busy:return
        if key!=self.key:self.key=key;self.count=0
        if self.count>=2 or (self.count and now-self.saved_at<.65):return
        self.busy=True;self.count+=1;self.saved_at=now
        slot=self.slot;self.slot=(slot+1)%6
        def save():
            try:
                self.root.mkdir(parents=True,exist_ok=True)
                frame,scale=self.picture(sample['frame'])
                name=f'transition-{slot}'
                cv2.imencode('.png',frame)[1].tofile(str(self.root/(name+'-target.png')))
                original=source.get('diagnostic_frame')
                if original is not None:cv2.imencode('.png',original)[1].tofile(str(self.root/(name+'-source.png')))
                clear=sample.get('clear_mask')
                if clear is not None:
                    clear=cv2.resize(clear,frame.shape[1::-1],interpolation=cv2.INTER_NEAREST)
                    cv2.imencode('.png',clear)[1].tofile(str(self.root/(name+'-clear.png')))
                point=source.get('player_hd')
                payload=dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),source=source['floor']['id'],
                    target=sample.get('selected_floor'),source_matrix=source['matrix'].tolist(),
                    source_zoom=source.get('zoom'),target_zoom=sample.get('zoom'),
                    player_hd=point.tolist() if point is not None else None,
                    source_scale=source.get('diagnostic_scale'),target_scale=scale,
                    has_source=original is not None,has_clear=clear is not None,
                    attempt=sample.get('shared_attempt'),source_at=source.get('sample_at'),target_at=sample['at'],
                    scope='Observed app capture, downscaled; projection accuracy requires separate verification')
                tmp=self.root/(name+'.tmp')
                tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),'utf8')
                tmp.replace(self.root/(name+'.json'))
            except (OSError,ValueError,cv2.error):pass
            finally:self.busy=False
        threading.Thread(target=save,daemon=True).start()
