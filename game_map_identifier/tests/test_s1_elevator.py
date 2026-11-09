"""A registered B1 and a separately checked B2 have distinct roles."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.s1_round_lock import RoundLock


def proof(at,layout='s1_13',kind='elevator',verified=False):
    return dict(at=at,floor=dict(id=layout+'_B2',floor='B2',layout_id=layout),
                unique_layout=True,proof_kind=kind,local_verified=True,elevator_verified=verified,
                matrix=np.eye(2,3),contradicts=verified)


def test_elevator_only_draws_this_floor_until_neighbours_distinguish_it():
    state=RoundLock()
    for t in (1.,2.,3.):
        state.observe(proof(t))
        assert state.region_layout=='s1_13' and not state.layout and state.blocks_floor('B3')
    state.observe(proof(4.,verified=True))
    assert state.layout=='s1_13' and state.elevator_lock and not state.blocks_floor('B3')


def test_first_checked_landing_and_duplicate_frame_do_not_require_three_votes():
    state=RoundLock()
    assert state.observe(proof(1.,verified=True)) and state.layout=='s1_13' and state.count==1
    state=RoundLock();state.observe(proof(1.))
    check=proof(1.,kind='full',verified=True)
    assert state.observe(check) and state.layout=='s1_13' and state.count==1
    assert not state.observe(check)


def test_rejected_elevator_does_not_consume_independent_capture():
    state=RoundLock();state.layout='s1_07';state.b1_conflict=True
    assert not state.observe(proof(1.))
    assert state.observe(proof(1.,kind='full',verified=True)) and state.layout=='s1_13'


def test_new_ambiguity_invalidates_unverified_elevator_only():
    state=RoundLock();state.observe(proof(1.));state.forget_elevator()
    assert state.elevator is None and not state.proposal and not state.region_layout
    state.observe(proof(2.,verified=True));state.forget_elevator();assert state.layout=='s1_13'
