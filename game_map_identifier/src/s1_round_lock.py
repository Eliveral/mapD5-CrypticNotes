"""The round's identity is independent of its current screen position."""
class LandingReview:
    """Independent B3 evidence. This never votes on the round's B2 identity."""
    def __init__(self):
        self.layout=None
        self.waiting=False
        self.since=-1.
        self.pending=None
        self.cells=set()
        self.reference=None
        self.at=-1.

    def observe(self,hit,b2):
        if not hit or hit['floor']['floor']!='B3' or (hit['at']<=self.at or hit['at']<self.since):return False
        self.at=hit['at']
        if hit.get('conflicting_models'):
            changed=not self.waiting or self.pending is not None
            self.waiting=True;self.pending=None;self.cells.clear()
            self.reference=None
            return changed
        target=hit['floor']['layout_id']
        if target==(self.layout or b2):
            changed=self.waiting or self.pending is not None
            self.waiting=False
            self.pending=None;self.cells.clear();self.reference=hit
            return changed
        cells=set(hit['cells'])
        if target!=self.pending:
            self.pending=target;self.cells=cells
            return True
        if len(cells-self.cells)<3 and not hit.get('disjoint_proof'):return False
        self.waiting=False
        self.layout=target;self.reference=hit
        self.pending=None;self.cells.clear()
        return True


class RoundLock:
    required=3

    def __init__(self):
        self.landing=LandingReview()
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
        self.recheck_b2_after_landing=False
        self.b2_arrival=False
        self.b2_since=-1.
        self.direct_b2=False

    def enter_b2(self,at,new_visit,previous):
        self.b2_arrival=True
        self.b2_since=at
        if new_visit:
            self.direct_b2=previous!='B1'
            self.proposal=None;self.proposal_count=0;self.cells.clear()
            if not self.layout:self.candidate=None;self.count=0

    def enter_b3(self,at,new_visit):
        if new_visit:
            prior=self.landing.layout
            self.landing=LandingReview()
            self.landing.layout=prior
        self.landing.waiting=True
        self.landing.since=at

    @property
    def challenger(self):
        if self.layout and self.proposal and self.proposal['floor']['layout_id']!=self.layout:
            return self.proposal['floor']['layout_id']
        return None

    def preferred(self,level):
        if level=='B1':return None
        if level=='B3' and self.landing.layout:return self.landing.layout
        if level=='B2' and self.proposal:return self.proposal['floor']['layout_id']
        if self.b1_conflict:return None
        return self.layout

    def blocks_floor(self,level):
        disputed=bool(self.uncertain or self.challenger or self.b1_conflict)
        if level=='B3':
            return self.landing.waiting or bool(self.landing.pending) or ((disputed or self.b2_arrival) and not self.landing.layout)
        return (disputed or self.recheck_b2_after_landing or self.b2_arrival) and level!='B1'

    def observe_landing(self,hit):
        review=self.landing
        prior=review.layout
        base=None if self.blocks_floor('B2') else self.layout
        changed=review.observe(hit,base)
        if review.layout and review.layout!=prior and self.layout and review.layout!=self.layout:
            self.recheck_b2_after_landing=True
        return changed

    def permits_overlay(self, level, layout, candidates):
        if level=='B1':
            # Independent B1 recognition can dispute a stale lock. Until B2
            # decides, only a common region including the lock can be drawn.
            return not self.layout or self.layout in candidates
        if self.blocks_floor(level):return False
        expected=self.preferred(level) if level=='B3' else self.layout
        return not expected or (layout==expected and all(v==expected for v in candidates))

    def observe(self,reference):
        if not reference:return False
        if reference['floor']['floor']!='B2' or not reference.get('unique_layout') or reference.get('provisional'):
            return False
        stamp=reference['at']
        if stamp<=self.last_capture or stamp<self.b2_since:return False
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
            rechecked=self.b1_conflict or self.recheck_b2_after_landing or self.b2_arrival
            self.b2_arrival=False
            self.b1_conflict=False
            self.recheck_b2_after_landing=False
            self.reference=reference
            self.proposal=None;self.proposal_count=0;self.cells.clear()
            return disputed or previously_uncertain or rechecked
        if self.layout and not reference.get('contradicts'):return False
        if not self.proposal or layout!=self.proposal['floor']['layout_id']:
            self.proposal_count=0;self.cells.clear()
            if self.layout and self.landing.layout!=layout:
                self.landing=LandingReview();self.landing.waiting=True
        cells=set(reference.get('cells',()))
        novel=cells-self.cells
        if reference.get('proof_kind')=='partial' and self.proposal_count and len(novel)<3:return False
        self.proposal=reference
        self.cells.update(cells);self.proposal_count+=1
        if not self.layout:self.candidate,self.count=layout,self.proposal_count
        if self.proposal_count==self.required:
            if self.layout and self.layout!=layout and self.landing.layout!=layout:
                require_b3=self.landing.waiting or self.b2_arrival or self.direct_b2
                self.landing=LandingReview()
                self.landing.waiting=require_b3
            self.recheck_b2_after_landing=False
            self.layout=layout
            self.b2_arrival=False
            self.b1_conflict=False
            self.reference=reference
            self.candidate,self.count=layout,self.required
            self.proposal=None;self.proposal_count=0;self.cells.clear()
        return True

    def status(self):
        if self.uncertain:return '负二的固定结构比对有冲突 · 等待更多墙线'
        if self.challenger:return f'负二与旧布局 {self.layout[3:]} 冲突 · 复核布局 {self.challenger[3:]}（{self.proposal_count}/{self.required}）'
        if self.landing.layout and self.recheck_b2_after_landing:
            return f'负三已确认 {self.landing.layout[3:]} · 回负二时重新核对'
        if self.b1_conflict:return '负一与此前布局不符 · 等待重新核对负二'
        if self.landing.pending:return '负三出现不同的固定结构 · 核对 '+self.landing.pending[3:]
        if self.landing.waiting:return '正在独立核对传送后的负三位置'
        if self.landing.layout and self.landing.layout!=self.layout:
            b2=self.layout[3:] if self.layout else '未确认'
            return f'负二 {b2} · 负三独立确认 {self.landing.layout[3:]}'
        if self.b2_arrival:return f'负二落点重新核对 · {self.proposal_count}/{self.required}'
        if self.layout:return '本局已锁定 · 布局 '+self.layout[3:]+'（负二已确认 3 次）'
        if self.candidate:return f'负二核对 {self.count}/{self.required} · 布局 {self.candidate[3:]}'
        return '等待负二确认本局布局'

