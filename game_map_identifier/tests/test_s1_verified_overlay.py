"""A last-round pose cannot certify the first pixels of a different map."""
from src.s1_round_lock import RoundLock, direct_pose
from src.s1_trigger_live import LiveController


def ref(layout,stamp,level,**kwargs):
    return dict(floor=dict(floor=level,layout_id=layout,id=layout+'_'+level),at=stamp,
                unique_layout=True,cells={(1,1),(2,2),(3,3)},**kwargs)


def test_old_b3_review_cannot_survive_contradictory_b2():
    lock=RoundLock();lock.layout='s1_03'
    lock.observe_landing(ref('s1_03',1,'B3',disjoint_proof=True))
    # Previously independently identified B3 (not just inherited from B2).
    lock.landing.layout='s1_03'
    lock.observe(ref('s1_12',4,'B2',conflicting_models=True))
    assert lock.blocks_floor('B3') and not lock.landing.reference
    assert not lock.observe_landing(ref('s1_03',3,'B3',disjoint_proof=True))
    assert not lock.permits_overlay('B3','s1_03',['s1_03'])
    lock.observe_landing(ref('s1_13',5,'B3'))
    lock.observe_landing(ref('s1_13',6,'B3'))
    assert lock.blocks_floor('B3')
    lock.observe_landing(ref('s1_13',7,'B3',disjoint_proof=True))
    assert lock.permits_overlay('B3','s1_13',['s1_13']) and lock.layout=='s1_03'


def test_entrance_interrupts_old_jobs_and_a_fresh_b2_restores_paired_passage():
    lock=RoundLock();lock.layout='s1_01';lock.landing.layout='s1_01'
    assert lock.question_lower_floors(10)
    assert lock.blocks_floor('B2') and lock.blocks_floor('B3')
    assert not lock.observe(ref('s1_01',8,'B2',proof_kind='full'))
    assert not lock.observe(ref('s1_01',11,'B2',proof_kind='partial'))
    lock.observe(ref('s1_01',12,'B2',proof_kind='full',local_verified=True))
    assert lock.permits_overlay('B3','s1_01',['s1_01'])


def test_b2_reconfirming_independent_b3_keeps_its_verified_pose():
    lock=RoundLock();lock.layout='s1_03'
    lock.observe_landing(ref('s1_13',2,'B3',disjoint_proof=True))
    lock.observe(ref('s1_13',3,'B2',proof_kind='partial',local_verified=True,contradicts=True))
    assert lock.landing.layout=='s1_13' and not lock.blocks_floor('B3')


def test_accept_only_linked_measurements_as_direct_poses():
    assert not direct_pose({'search':'fixed_walls'})
    assert not direct_pose({'search':'paired_corridor'},paired=True)
    assert not direct_pose({'search':'paired_player'})
    assert direct_pose({'search':'paired_player'},paired=True)
    assert direct_pose({'search':'camera_zoom'})
    assert direct_pose({'search':'fixed_anvil','device_kind':'anvil'})


def test_rejected_fixed_walls_cannot_seed_the_pan_cache():
    from unittest.mock import Mock
    c=LiveController.__new__(LiveController)
    c.round_lock=RoundLock();c.round_lock.layout='s1_01';c.app=Mock()
    c.app.layout_limit.get.return_value='自动'
    candidate=ref('s1_01',2,'B3',search='fixed_walls')
    assert not c.has_current_pose({'selected_floor':'B3'},candidate)
