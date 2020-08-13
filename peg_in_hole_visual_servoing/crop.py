import argparse
import json
from typing import List
from functools import partial

import cv2
import numpy as np

import rospy
import sensor_msgs.msg
from ros_numpy.image import numpy_to_image, image_to_numpy

import peg_in_hole_visual_servoing_api.srv
from . import servo_configure


def configure(servo_config):
    _configure = rospy.ServiceProxy('/servo/configure', peg_in_hole_visual_servoing_api.srv.SetString)
    _configure.wait_for_service()
    return _configure(json.dumps(servo_config))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='plastic')
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--crop-hole-scale', type=float, default=5.)

    args = parser.parse_args()
    crop_size = args.crop_size
    crop_hole_scale = args.crop_hole_scale

    rospy.init_node('servo_crop_node')
    subs = []  # type: List[rospy.Subscriber]
    pubs = []  # type: List[rospy.Publisher]
    topics = []
    rect_maps = []

    # Reusing pub-subs is a little messy.
    # I tried a cleaner approach that would always unregister and register pubsubs on configure,
    # but there was a delay of approx 2 seconds.

    def _configure(msg: peg_in_hole_visual_servoing_api.srv.SetStringRequest):
        crop_configs = json.loads(msg.str)['crop_configs']
        n = len(crop_configs)
        if n != len(subs):
            for sub in subs:
                sub.unregister()
            for pub in pubs:
                pub.unregister()
            subs.clear()
            pubs.clear()
            topics.clear()
            rect_maps.clear()

            for i in range(n):
                pubs.append(rospy.Publisher('/servo/crop_{}'.format(i), sensor_msgs.msg.Image, queue_size=1))

        for i, crop_config in enumerate(crop_configs):
            image_topic, K, dist_coeffs, hole_center, hole_size, hole_normal = (crop_config[key] for key in (
                'image_topic', 'K', 'dist_coeffs', 'hole_center', 'hole_size', 'hole_normal'
            ))

            K, dist_coeffs = np.array(K), np.array(dist_coeffs)
            roi_transform = servo_configure.get_roi_transform(crop_config, crop_hole_scale, crop_size)
            roi_K = roi_transform @ K
            rect_map = cv2.initUndistortRectifyMap(K, dist_coeffs, np.eye(3), roi_K,
                                                   (crop_size, crop_size), cv2.CV_32FC1)
            if i < len(rect_maps):
                rect_maps[i] = rect_map
            else:
                rect_maps.append(rect_map)

            if i >= len(subs) or topics[i] != image_topic:
                def cb(img_msg: sensor_msgs.msg.Image, i):
                    img = image_to_numpy(img_msg)
                    crop = cv2.remap(img, *rect_maps[i], cv2.INTER_LINEAR)
                    crop_msg = numpy_to_image(crop, 'rgb8')
                    crop_msg.header.stamp = img_msg.header.stamp
                    pubs[i].publish(crop_msg)

                sub = rospy.Subscriber(
                    image_topic, sensor_msgs.msg.Image, partial(cb, i=i),
                    queue_size=1, buff_size=1920 * 1080 * 3 * 8 + 2 ** 16
                )
                if i < len(subs):
                    subs[i].unregister()
                    subs[i] = sub
                    topics[i] = image_topic
                else:
                    subs.append(sub)
                    topics.append(image_topic)

        return True

    rospy.Service('/servo/configure', peg_in_hole_visual_servoing_api.srv.SetString, _configure)
    rospy.spin()


if __name__ == '__main__':
    main()
