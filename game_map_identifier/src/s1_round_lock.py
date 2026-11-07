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
        self.proposal=None
        self.proposal_count=0
        self.cells=set()
        self.uncertain=False
        self.b1_conflict=False

    @property
    def challenger(self):
        if self.layout and self.proposal and self.proposal['floor']['layout_id']!=self.layout:
            return self.proposal['floor']['layout_id']
        return None

    def preferred(self,level):
        if level=='B1':return None
        if level=='B2' and self.proposal:return self.proposal['floor']['layout_id']
        if self.b1_conflict:return None
        return self.layout

    def permits_overlay(self, level, layout, candidates):
        if level=='B1':
            # Independent B1 recognition can dispute a stale lock. Until B2
            # decides, only a common region including the lock can be drawn.
            return not self.layout or self.layout in candidates
        if self.uncertain or self.challenger or self.b1_conflict:
            return False
        return not self.layout or (layout==self.layout and all(v==self.layout for v in candidates))

    def observe(self,reference):
        if not reference:return False
        if reference['floor']['floor']!='B2' or not reference.get('unique_layout') or reference.get('provisional'):
            return False
        stamp=reference['at']
        if stamp<=self.last_capture:return False
        self.last_capture=stamp
        if reference.get('conflicting_models'):
            changed=not self.uncertain
            self.uncertain=True
            self.proposal=None;self.proposal_count=0;self.cells.clear()
            return changed
        previously_uncertain=self.uncertain
        self.uncertain=False
        layout=reference['floor']['layout_id']
        if self.layout==layout:
            disputed=bool(self.challenger)
            rechecked=self.b1_conflict
            self.b1_conflict=False
            self.reference=reference
            self.proposal=None;self.proposal_count=0;self.cells.clear()
            return disputed or previously_uncertain or rechecked
        if self.layout and not reference.get('contradicts'):return False
        if not self.proposal or layout!=self.proposal['floor']['layout_id']:
            self.proposal_count=0;self.cells.clear()
        cells=set(reference.get('cells',()))
        novel=cells-self.cells
        if reference.get('proof_kind')=='partial' and self.proposal_count and len(novel)<3:return False
        self.proposal=reference
        self.cells.update(cells);self.proposal_count+=1
        if not self.layout:self.candidate,self.count=layout,self.proposal_count
        if self.proposal_count==self.required:
            self.layout=layout
            self.b1_conflict=False
            self.reference=reference
            self.candidate,self.count=layout,self.required
            self.proposal=None;self.proposal_count=0;self.cells.clear()
        return True

    def status(self):
        if self.uncertain:return '负二的固定结构比对有冲突 · 等待更多墙线'
        if self.challenger:return f'负二与旧布局 {self.layout[3:]} 冲突 · 复核布局 {self.challenger[3:]}（{self.proposal_count}/{self.required}）'
        if self.b1_conflict:return '负一与此前布局不符 · 等待重新核对负二'
        if self.layout:return '本局已锁定 · 布局 '+self.layout[3:]+'（负二已确认 3 次）'
        if self.candidate:return f'负二核对 {self.count}/{self.required} · 布局 {self.candidate[3:]}'
        return '等待负二确认本局布局'

