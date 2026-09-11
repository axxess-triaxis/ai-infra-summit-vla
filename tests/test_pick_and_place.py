"""Real physics pick-and-place, end to end, no mocks.

Uses the cup (radially symmetric) rather than the fork/knife: our IK is
position-only (see control/ik.py), so grasp success for a thin, elongated
object depends on approaching along its narrow axis by luck of whatever
orientation the damped-least-squares solve converges to. A cylinder can be
grasped from any horizontal approach direction, so it's the reliable smoke
test; making the fork/knife graspable needs position+orientation IK, tracked
as a follow-up, not a blocker for the online-track submission.
"""

import numpy as np

from aisummit.control.primitives import pick, place
from aisummit.sim.env import DinnerTableEnv


def test_right_arm_picks_and_places_cup():
    env = DinnerTableEnv()
    try:
        env.reset(randomize=False)
        start_pos = env.body_xpos("cup").copy()

        ctrl = env.current_ctrl()
        ctrl, obs = pick(env, ctrl, side="right", object_name="cup")
        lifted_pos = obs.object_positions["cup"]
        assert lifted_pos[2] > start_pos[2] + 0.05, "cup was not lifted off the table"

        target = start_pos + np.array([-0.05, 0.05, 0.0])
        ctrl, obs = place(env, ctrl, side="right", target_xyz=target)
        final_pos = obs.object_positions["cup"]

        assert np.linalg.norm(final_pos[:2] - target[:2]) < 0.06, (
            f"cup placed at {final_pos[:2]}, expected near {target[:2]}"
        )
        assert final_pos[2] < start_pos[2] + 0.05, "cup should have been set back down"
    finally:
        env.close()
