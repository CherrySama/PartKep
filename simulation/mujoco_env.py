"""
simulation/mujoco_env.py
Created by Yinghao Ho on 2026-04-19                    : reset to home configuration
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
from typing import List, Dict
from scipy.spatial.transform import Rotation

Q_HOME = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])

GRIPPER_OPEN  = 255.0
GRIPPER_CLOSE = 0.0

STEPS_PER_WAYPOINT  = 30      # physics steps advanced per waypoint
SLEEP_PER_WAYPOINT  = 0.01   # sleep (s) after each waypoint
WELD_SETTLE_STEPS   = 50     # physics steps after weld activation


class MuJoCoEnv:
    """
    Franka Panda MuJoCo execution environment.

    Args:
        scene_xml : path to scene.xml
        n_steps   : interpolation steps for move_to (debug)
        step_dt   : sleep (s) per step in move_to
    """

    def __init__(
        self,
        scene_xml: str   = "assets/franka_emika_panda/scene.xml",
        n_steps:   int   = 150,
        step_dt:   float = 0.005,
    ):
        self.model   = mujoco.MjModel.from_xml_path(scene_xml)
        self.data    = mujoco.MjData(self.model)
        self.n_steps = n_steps
        self.step_dt = step_dt

        self._reset_state()

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance  = 2.0
        self.viewer.cam.azimuth   = 135
        self.viewer.cam.elevation = -20

        print(f"[MuJoCoEnv] home config: {np.round(Q_HOME, 4)}")

    def execute_pick_place(self, segments: List[Dict]):
        """
        Execute a full pick-place segment sequence.

        post_actions triggered after each segment:
            'close_gripper'   -> close gripper
            'activate_weld'   -> activate cup_weld + settle WELD_SETTLE_STEPS steps
            'open_gripper'    -> open gripper
            'deactivate_weld' -> deactivate cup_weld

        Args:
            segments: output of MotionPlanner.plan_pick_place(),
                      List[Dict] with keys 'label', 'waypoints', 'post_actions'
        """
        total = sum(len(s['waypoints']) for s in segments)
        print(f"[execute_pick_place] {len(segments)} segments, {total} waypoints")

        for seg in segments:
            label        = seg['label']
            waypoints    = seg['waypoints']
            post_actions = seg['post_actions']

            print(f"  [{label}] {len(waypoints)} waypoints")
            for q in waypoints:
                self.data.ctrl[:7] = q
                for _ in range(STEPS_PER_WAYPOINT):
                    mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                time.sleep(SLEEP_PER_WAYPOINT)

            # trigger post_actions after segment completes
            for action in post_actions:
                if action == 'close_gripper':
                    self._set_gripper(open=False)
                elif action == 'activate_weld':
                    self._set_weld(active=True)
                    print(f"  [weld] activated, settling {WELD_SETTLE_STEPS} steps")
                    self._settle(WELD_SETTLE_STEPS)
                elif action == 'open_gripper':
                    self._set_gripper(open=True)
                elif action == 'deactivate_weld':
                    self._set_weld(active=False)
                else:
                    print(f"  [warning] unknown post_action: {action}")

    def execute_trajectory(self, waypoints: List[np.ndarray], gripper_ctrl: float = GRIPPER_OPEN):
        """
        Execute flat waypoints (legacy, gripper stays open throughout).

        Args:
            waypoints   : List[np.ndarray], each shape=(7,)
            gripper_ctrl: gripper control value, default GRIPPER_OPEN
        """
        print(f"[execute_trajectory] {len(waypoints)} waypoints")
        for q in waypoints:
            self.data.ctrl[:7] = q
            self.data.ctrl[7]  = gripper_ctrl
            for _ in range(STEPS_PER_WAYPOINT):
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(SLEEP_PER_WAYPOINT)

    def get_site_xpos(self, site_name: str) -> np.ndarray:
        """Return world position of the named site."""
        mujoco.mj_forward(self.model, self.data)
        site_id = self.model.site(site_name).id
        return self.data.site_xpos[site_id].copy()

    def reset(self):
        """Reset to home configuration and restore cup position."""
        self._reset_state()
        self.move_to(Q_HOME, label="home")

    def close(self):
        """Close the viewer."""
        self.viewer.close()

    def _set_gripper(self, open: bool):
        self.data.ctrl[7] = GRIPPER_OPEN if open else GRIPPER_CLOSE
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(SLEEP_PER_WAYPOINT)

    def _set_weld(self, active: bool):
        weld_id = self.model.equality('cup_weld').id
        if active:
            cup_id  = self.model.body('cup').id
            hand_id = self.model.body('hand').id

            pos_cup  = self.data.xpos[cup_id].copy()
            pos_hand = self.data.xpos[hand_id].copy()

            # xquat: MuJoCo native [w,x,y,z], guaranteed unit quaternion
            quat_cup  = self.data.xquat[cup_id].copy()   # [w,x,y,z]
            quat_hand = self.data.xquat[hand_id].copy()  # [w,x,y,z]

            # convert to scipy [x,y,z,w]
            rot_cup  = Rotation.from_quat([quat_cup[1],  quat_cup[2],  quat_cup[3],  quat_cup[0]])
            rot_hand = Rotation.from_quat([quat_hand[1], quat_hand[2], quat_hand[3], quat_hand[0]])

            # hand origin in cup local frame
            rel_pos = rot_cup.inv().apply(pos_hand - pos_cup)

            # hand rotation relative to cup
            xyzw = (rot_cup.inv() * rot_hand).as_quat()  # [x,y,z,w]
            wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # MuJoCo [w,x,y,z]

            self.model.eq_data[weld_id, 0:3]  = 0.0      # anchor
            self.model.eq_data[weld_id, 3:6]  = rel_pos  # relative position
            self.model.eq_data[weld_id, 6:10] = wxyz     # relative quaternion

            print(f"  [weld] rel_pos  : {np.round(self.model.eq_data[weld_id, 3:6], 4)}")
            print(f"  [weld] rel_quat : {np.round(self.model.eq_data[weld_id, 6:10], 4)}")

        self.data.eq_active[weld_id] = active
        mujoco.mj_forward(self.model, self.data)

    def _settle(self, n_steps: int):
        """
        Advance physics n_steps with ctrl unchanged, allowing constraints to stabilize.

        Args:
            n_steps: number of steps to settle (recommended: 50)
        """
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def move_to(self, q_target: np.ndarray, label: str = ""):
        """
        Linear interpolation in joint space to q_target (debug / reset only).

        Args:
            q_target: target joint angles, shape=(7,)
            label:    log label
        """
        if label:
            print(f"[move_to] {label}")

        q_start = self.data.qpos[:7].copy()
        for i in range(self.n_steps + 1):
            alpha               = i / self.n_steps
            q_interp            = (1 - alpha) * q_start + alpha * q_target
            self.data.ctrl[:7]  = q_interp
            self.data.ctrl[7]   = GRIPPER_OPEN
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(self.step_dt)

    def _reset_state(self):
        """Reset simulation: arm from keyframe 0, cup restored to initial pose."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._reset_cup()
        mujoco.mj_forward(self.model, self.data)

    def _reset_cup(self):
        """
        Restore cup freejoint to initial pose.
        cup body pos="0.5 0.05 0.45", identity quaternion.
        """
        try:
            jnt_id = self.model.joint('cup_free').id
            adr    = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[adr + 0] = 0.5
            self.data.qpos[adr + 1] = 0.05
            self.data.qpos[adr + 2] = 0.45
            self.data.qpos[adr + 3] = 1.0
            self.data.qpos[adr + 4] = 0.0
            self.data.qpos[adr + 5] = 0.0
            self.data.qpos[adr + 6] = 0.0
        except Exception:
            pass