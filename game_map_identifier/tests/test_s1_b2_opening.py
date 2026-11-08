"""A verified local B2 pose can be displayed before the round is locked."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.s1_round_lock import RoundLock


def partial(at,layout='s1_05',strong=True,**extra):
    return dict(at=at,floor=dict(layout_id=layout,floor='B2'),unique_layout=True,
                region_verified=strong,proof_kind='partial',cells={(1,2),(3,4),(5,6)},
                matrix=np.eye(2,3),**extra)


def test_first_unique_region_displays_only_on_b2():
    lock=RoundLock();lock.observe_opening('B2',0)
    lock.observe(partial(1))
    for t in (2,3,4):
        lock.observe(partial(t))
    assert lock.count==1 and lock.layout is None
    assert lock.permits_overlay('B2','s1_05',['s1_05'])
    assert not lock.permits_overlay('B2','s1_01',['s1_01'])
    assert not lock.permits_overlay('B3','s1_05',['s1_05'])


def test_ambiguous_or_changed_region_revokes_local_display():
    lock=RoundLock();lock.observe_opening('B2',0)
    lock.observe(partial(1,strong=False))
    assert lock.blocks_floor('B2')
    lock.observe(partial(2));assert not lock.blocks_floor('B2')
    lock.observe(partial(3,conflicting_models=True));assert lock.blocks_floor('B2')
    lock.observe(partial(4));lock.observe(partial(5,'s1_01',strong=False))
    assert lock.blocks_floor('B2') and not lock.region_layout


def test_partial_region_cannot_displace_existing_round_by_repetition():
    lock=RoundLock();lock.layout='s1_09';lock.observe_opening('B2',0)
    for t in range(1,5):lock.observe(partial(t,contradicts=True,camera_support=True))
    assert lock.layout=='s1_09' and lock.proposal_count==1 and lock.blocks_floor('B2')


def test_corroborated_separate_cameras_can_complete_cold_lock():
    lock=RoundLock();lock.observe_opening('B2',0)
    lock.observe(partial(1));lock.observe(partial(2,camera_support=True))
    assert not lock.permits_overlay('B3','s1_05',['s1_05'])
    lock.observe(partial(3,camera_support=True))
    assert lock.layout=='s1_05' and lock.permits_overlay('B3','s1_05',['s1_05'])
