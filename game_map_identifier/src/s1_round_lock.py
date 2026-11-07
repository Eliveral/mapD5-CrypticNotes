"""The round's identity is independent of its current screen position."""
class RoundLock:
    required=3

    def __init__(self):
        self.owner=None
        self.candidate=None
        self.count=0
        self.last_capture=-1.
        self.layout=None
        self.reference=None

    def observe(self,reference):
        if self.layout or not reference:return False
        if reference['floor']['floor']!='B2' or not reference.get('unique_layout') or reference.get('provisional'):
            return False
        stamp=reference['at']
        if stamp<=self.last_capture:return False
        self.last_capture=stamp
        layout=reference['floor']['layout_id']
        if layout!=self.candidate:
            self.candidate=layout
            self.count=0
        self.count+=1
        if self.count==self.required:
            self.layout=layout
            self.reference=reference
        return True

    def status(self):
        if self.layout:return '本局已锁定 · 布局 '+self.layout[3:]+'（负二已确认 3 次）'
        if self.candidate:return f'负二核对 {self.count}/{self.required} · 布局 {self.candidate[3:]}'
        return '等待负二确认本局布局'

