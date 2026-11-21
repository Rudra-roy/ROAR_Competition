import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import gate  # noqa: E402


def R(status="finished", coll=0, clearance=0.2):
    return dict(status=status, coll=coll, clearance=clearance)


def test_pass_needs_every_race_clean_and_the_closest_approach_over_the_limit():
    ok, text = gate.verdict([R(clearance=0.31), R(clearance=0.12), R(clearance=0.18)], 0.10)
    assert ok and "3/3" in text and "0.12" in text
    assert not gate.verdict([R(clearance=0.31), R(clearance=0.09)], 0.10)[0]  # one race too close to a wall
    assert not gate.verdict([R(), R(status="aborted", coll=1, clearance=None)], 0.10)[0]  # a collision
    assert not gate.verdict([R(), R(status="crashed", coll=0, clearance=None)], 0.10)[0]  # a client crash is not clean


def test_a_finished_race_without_a_clearance_cannot_pass_a_wall_gate():
    assert not gate.verdict([R(clearance=0.3), R(clearance=None)], 0.10)[0]
    assert gate.verdict([R(clearance=None), R(clearance=None)], 0.10, need_walls=False)[0]  # no map: time/collisions only
