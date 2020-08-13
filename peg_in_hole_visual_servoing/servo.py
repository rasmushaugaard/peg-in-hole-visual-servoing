import time
from typing import List, Sequence, Union
from functools import partial

import numpy as np
from transform3d import Transform, SceneNode, SceneState
from ur_control import Robot, DeviatingMotionError

import rospy
from std_msgs.msg import Float64MultiArray

from .servo_configure import get_roi_transform
from . import utils
from . import crop


def servo(peg_robot: Robot, peg_tcp_node: SceneNode, scene_state: SceneState,
          servo_config: dict, camera_nodes: Sequence[SceneNode],
          aux_robots: Sequence[Robot] = (), aux_tcp_nodes: Sequence[SceneNode] = (),
          insertion_direction_tcp=np.array((0, 0, 1)),
          err_tolerance_scale=0.05, timeout=5.,
          max_travel: float = None, max_travel_scale=3.,
          alpha_target=0.9, alpha_err=0.9):
    state = scene_state.copy()
    crop_configs = servo_config['crop_configs']
    insertion_direction_tcp = np.asarray(insertion_direction_tcp) / np.linalg.norm(insertion_direction_tcp)
    assert len(aux_robots) == len(aux_tcp_nodes)
    tcp_nodes = (peg_tcp_node, *aux_tcp_nodes)
    robots = (peg_robot, *aux_robots)
    assert len(servo_config['robots_q_init']) == len(robots)
    assert len(aux_robots)
    n_cams = len(crop_configs)
    z_ests, diameter_est = servo_config['z_ests'], servo_config['diameter_est']
    assert n_cams == len(z_ests) == len(camera_nodes)
    err_tolerance = diameter_est * err_tolerance_scale
    if max_travel is None:
        max_travel = diameter_est * max_travel_scale
    assert crop.configure(servo_config)

    crop_K_invs = [np.linalg.inv(get_roi_transform(crop_config) @ crop_config['K']) for crop_config in crop_configs]
    cams_points = [None for _ in range(n_cams)]  # type: List[Union[None, (float, np.ndarray)]]
    new_data = [False]

    def handle_points(msg: Float64MultiArray, cam_idx: int):
        timestamp, points = msg.data[0], np.array(msg.data[1:]).reshape(2, 2)
        cams_points[cam_idx] = timestamp, points
        new_data[0] = True

    subs = []
    for cam_idx in range(n_cams):
        subs.append(rospy.Subscriber(
            '/servo/crop_{}/points'.format(cam_idx), Float64MultiArray,
            partial(handle_points, cam_idx=cam_idx), queue_size=1
        ))

    scene_configs_times = []  # type: List[float]
    scene_configs = []  # type: List[Sequence[Transform]]

    def add_current_scene_config():
        scene_configs.append([r.base_t_tcp() for r in robots])
        scene_configs_times.append(time.time())

    def update_scene_state(timestamp):
        transforms = scene_configs[utils.bisect_closest(scene_configs_times, timestamp)]
        for tcp_node, transform in zip(tcp_nodes, transforms):
            state[tcp_node] = transform

    add_current_scene_config()

    peg_tcp_init_node = SceneNode(parent=peg_tcp_node.parent)
    state[peg_tcp_init_node] = scene_configs[-1][0]
    peg_tcp_cur_node = SceneNode(parent=peg_tcp_node.parent)

    base_p_tcp_rolling = state[peg_tcp_init_node].p
    err_rolling, err_size_rolling = None, err_tolerance * 10
    start = time.time()
    try:
        while err_size_rolling > err_tolerance:
            loop_start = time.time()
            add_current_scene_config()
            if new_data[0]:
                new_data[0] = False
                state[peg_tcp_cur_node] = scene_configs[-1][0]

                peg_tcp_init_t_peg_tcp = peg_tcp_init_node.t(peg_tcp_node, state)
                if np.linalg.norm(peg_tcp_init_t_peg_tcp.p) > max_travel:
                    raise DeviatingMotionError()
                if rospy.is_shutdown():
                    raise RuntimeError()
                if loop_start - start > timeout:
                    raise TimeoutError()

                # TODO: check for age of points
                # TODO: handle no camera inputs (raise appropriate error)
                move_dirs = []
                move_errs = []
                for cam_points, K_inv, cam_node, z_est in zip(cams_points, crop_K_invs, camera_nodes, z_ests):
                    if cam_points is None:
                        continue
                    timestamp, cam_points = cam_points
                    update_scene_state(timestamp)
                    peg_tcp_t_cam = peg_tcp_node.t(cam_node, state)

                    pts_peg_tcp = []
                    for p_img in cam_points:
                        p_cam = K_inv @ (*p_img, 1)
                        p_cam *= z_est / p_cam[2]
                        pts_peg_tcp.append(peg_tcp_t_cam @ p_cam)
                    hole_peg_tcp, peg_peg_tcp = pts_peg_tcp

                    view_dir = (peg_peg_tcp + hole_peg_tcp) / 2 - peg_tcp_t_cam.p
                    move_dir = np.cross(view_dir, insertion_direction_tcp)
                    move_dir /= np.linalg.norm(move_dir)

                    already_moved = state[peg_tcp_init_node].inv.rotate(
                        state[peg_tcp_cur_node].p - state[peg_tcp_node].p
                    )
                    move_err = np.dot(move_dir, (hole_peg_tcp - peg_peg_tcp) - already_moved)

                    move_dirs.append(move_dir)
                    move_errs.append(move_err)

                move_dirs = np.array(move_dirs)
                move_errs = np.array(move_errs)
                if len(move_dirs) > 0:
                    err_tcp, *_ = np.linalg.lstsq(move_dirs, move_errs, rcond=None)
                    if err_rolling is None:
                        err_rolling = err_tcp
                    err_rolling = alpha_err * err_rolling + (1 - alpha_err) * err_tcp
                    err_size_rolling = alpha_err * err_size_rolling + (1 - alpha_err) * np.linalg.norm(err_rolling)
                    base_t_tcp_target = state[peg_tcp_cur_node] @ Transform(p=err_tcp)
                    base_p_tcp_rolling = alpha_target * base_p_tcp_rolling + (1 - alpha_target) * base_t_tcp_target.p
            peg_robot.ctrl.servoL(
                (*base_p_tcp_rolling, *state[peg_tcp_init_node].rotvec),
                0.5, 0.25, peg_robot.dt, 0.2, 300
            )
            loop_duration = time.time() - loop_start
            time.sleep(max(0., peg_robot.dt - loop_duration))
    finally:
        peg_robot.ctrl.servoStop()
        for sub in subs:
            sub.unregister()
