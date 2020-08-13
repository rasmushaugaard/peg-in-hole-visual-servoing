from typing import Sequence

import numpy as np
import cv2
from transform3d import SceneNode, SceneState
from ur_control import Robot

import rospy
import sensor_msgs.msg
from ros_numpy.image import image_to_numpy

from . import utils


def get_z_ests(servo_config, cam_nodes: Sequence[SceneNode], state: SceneState, resolution=224):
    crop_Ks = [get_roi_transform(crop_config) @ crop_config['K'] for crop_config in servo_config['crop_configs']]

    frame_node = cam_nodes[0]
    line_points = []
    line_directions = []

    for K, cam_node in zip(crop_Ks, cam_nodes):
        K_inv = np.linalg.inv(K)
        dir_cam = K_inv @ (resolution / 2, resolution / 3, 1)
        frame_t_cam = frame_node.t(cam_node, state)
        line_directions.append(frame_t_cam.rotate(dir_cam))
        line_points.append(frame_t_cam.p)

    p = utils.closest_point_to_lines(np.array(line_points), np.array(line_directions))
    z_ests = [(cam_node.t(frame_node, state) @ p)[2] for cam_node in cam_nodes]
    return z_ests


def get_diameter_est(servo_config, zs):
    diameter_ests = []
    for crop_config, z in zip(servo_config['crop_configs'], zs):
        hole_size_px = crop_config['hole_size']
        K = np.array(crop_config['K'])
        f = np.sqrt(np.linalg.det(K[:2, :2]))
        diameter_ests.append(hole_size_px * z / f)
    return np.mean(diameter_ests)


def get_roi_transform(crop_config, crop_hole_scale=5., crop_size=224):
    hole_center, hole_size, hole_normal = (crop_config[key] for key in ('hole_center', 'hole_size', 'hole_normal'))
    angle = -np.arctan2(hole_normal[1], hole_normal[0]) + np.pi / 2
    S, C = np.sin(angle), np.cos(angle)
    M = np.eye(3)
    M[:2, :2] = np.array(((C, -S), (S, C)))
    size = hole_size * crop_hole_scale
    M[:2, 2] = (M[:2, :2] @ -np.array(hole_center)) + (size / 2, size / 3)
    M[:2] *= crop_size / size
    return M


# TODO: from known peg-tcp position

def config_from_demonstration(
        peg_robot: Robot, aux_robots: Sequence[Robot],
        peg_tcp_node: SceneNode, aux_tcp_nodes: Sequence[SceneNode],
        scene_state: SceneState,
        image_topics: Sequence[str], camera_nodes: Sequence[SceneNode],
        Ks: Sequence[np.ndarray], dist_coeffs: Sequence[np.ndarray],
        diameter_est: float = None,
):
    n_cams = len(camera_nodes)
    assert n_cams == len(Ks) == len(dist_coeffs) == len(image_topics)
    robots = (peg_robot, *aux_robots)
    tcp_nodes = (peg_tcp_node, *aux_tcp_nodes)
    assert len(robots) == len(tcp_nodes)
    for r in robots:
        r.ctrl.teachMode()
    input('move robots into start position for visual servoing and press enter')
    for r in robots:
        r.ctrl.endTeachMode()

    robots_q_init = [r.recv.getActualQ() for r in robots]
    robots_t_init = [r.base_t_tcp() for r in robots]
    state = scene_state.copy()

    for base_t_tcp, tcp_node in zip(robots_t_init, tcp_nodes):
        state[tcp_node] = base_t_tcp

    crop_configs = []
    for image_topic, K, dist_coeff in zip(image_topics, Ks, dist_coeffs):
        K, dist_coeff = np.array(K), np.array(dist_coeff)
        rect_maps = cv2.initUndistortRectifyMap(K, dist_coeff, np.eye(3), K, (1920, 1080), cv2.CV_32FC1)
        img = rospy.wait_for_message(image_topic, sensor_msgs.msg.Image, timeout=1)
        img = image_to_numpy(img)
        img = cv2.remap(img, *rect_maps, cv2.INTER_LINEAR)
        hole_points = utils.gui_select_vector('mark hole (longest line within the hole)', lambda: img)
        hole_center = np.mean(hole_points, axis=0).round().astype(int)
        hole_size = int(np.round(np.linalg.norm(hole_points[1] - hole_points[0])))
        hole_points = utils.gui_select_vector('draw vector from hole towards peg along insertion direction',
                                              lambda: img, arrow=True, first_point=hole_center)
        hole_normal = hole_points[1] - hole_points[0]
        crop_configs.append({
            'image_topic': image_topic,
            'K': K.tolist(),
            'dist_coeffs': dist_coeff.tolist(),
            'hole_center': hole_center.tolist(),
            'hole_size': hole_size,
            'hole_normal': hole_normal.tolist()
        })

    servo_config = {
        'robots_q_init': [list(q) for q in robots_q_init],
        'robots_t_init': [list(t.xyz_rotvec) for t in robots_t_init],
        'crop_configs': crop_configs,
    }

    z_ests = get_z_ests(servo_config, camera_nodes, state)
    diameter_est = get_diameter_est(servo_config, z_ests) if diameter_est is None else diameter_est
    servo_config['z_ests'] = z_ests
    servo_config['diameter_est'] = diameter_est

    return servo_config
