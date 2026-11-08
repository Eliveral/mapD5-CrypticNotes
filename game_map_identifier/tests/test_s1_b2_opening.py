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


def test_replacement_visible_on_b2_without_promoting_round():
    lock=RoundLock();lock.layout='s1_14';lock.b1_conflict=True
    proof=partial(1,'s1_11',strong=False,contradicts=True,local_verified=True)
    assert lock.observe(proof)
    assert lock.proposal_count==1 and lock.layout=='s1_14'
    assert lock.permits_overlay('B2','s1_11',['s1_11'])
    assert not lock.permits_overlay('B2','s1_14',['s1_14'])
    assert not lock.permits_overlay('B3','s1_11',['s1_11'])
    assert not lock.permits_overlay('B3','s1_14',['s1_14'])
    lock.observe(partial(2,'s1_12',conflicting_models=True))
    assert lock.blocks_floor('B2') and lock.region_layout is None
    lock.observe(partial(3,'s1_12',contradicts=False))
    assert lock.blocks_floor('B2') and lock.layout=='s1_14'


def test_distinct_strong_regions_can_grant_only_local_replacement():
    lock=RoundLock();lock.layout='s1_14'
    p=partial(1,'s1_11',contradicts=True)
    lock.observe(p);lock.observe(dict(p,at=2))
    assert lock.blocks_floor('B2') and lock.proposal_count==1
    lock.observe(dict(p,at=3,cells={(9,1),(10,3),(14,2),(15,4)}))
    assert lock.region_layout=='s1_11' and lock.layout=='s1_14'
    assert not lock.blocks_floor('B2') and lock.blocks_floor('B3')
    lock.observe(partial(4,'s1_14',strong=True))
    assert lock.region_layout is None and lock.permits_overlay('B2','s1_14',['s1_14'])


def test_new_b1_to_b2_local_evidence_does_not_authorize_b3():
    lock=RoundLock();lock.observe_opening('B1',0)
    lock.observe(partial(1,'s1_11',strong=True))
    assert lock.permits_overlay('B2','s1_11',['s1_11']) and lock.blocks_floor('B3')
    lock.observe(partial(2,'s1_12',strong=False))
    assert lock.region_layout is None and lock.blocks_floor('B2') and lock.blocks_floor('B3')


def test_verified_b1_can_draw_locally_without_changing_the_round():
    lock=RoundLock();lock.layout='s1_11'
    assert lock.permits_overlay('B1','s1_13',['s1_13','s1_02'])
    assert not lock.permits_overlay('B1','s1_11',['s1_13','s1_02'])
    assert lock.layout=='s1_11'


def test_new_b2_releases_an_inherited_b3_from_an_older_round():
    lock=RoundLock();lock.layout='s1_11'
    assert not lock.observe_landing(dict(partial(1,'s1_11'),floor={'floor':'B3','layout_id':'s1_11'}))
    assert lock.landing.reference is not None
    for stamp in (2,3,4):
        lock.observe(dict(partial(stamp,'s1_13',contradicts=True),proof_kind='full'))
    assert lock.layout=='s1_13' and not lock.blocks_floor('B3') and lock.landing.reference is None


def test_contradictory_current_b3_or_fresh_inherited_observation_keeps_review():
    for contradictory in (True,False):
        lock=RoundLock();lock.layout='s1_11'
        lock.observe(dict(partial(2,'s1_13',contradicts=True),proof_kind='full'))
        lock.observe_landing(dict(partial(2.5,'s1_11'),floor={'floor':'B3','layout_id':'s1_11'},
                                  conflicting_models=contradictory))
        for stamp in (3,4):
            lock.observe(dict(partial(stamp,'s1_13',contradicts=True),proof_kind='full'))
        assert lock.layout=='s1_13' and lock.blocks_floor('B3')


def test_b3_direct_start_still_waits_until_b2_is_fully_confirmed():
    lock=RoundLock();lock.observe_opening('B3',1)
    for t in (2,3):lock.observe(dict(partial(t,'s1_13'),proof_kind='full'))
    assert lock.blocks_floor('B3')
    lock.observe(dict(partial(4,'s1_13'),proof_kind='full'))
    assert not lock.blocks_floor('B3')
