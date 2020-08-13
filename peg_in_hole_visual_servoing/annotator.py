import argparse
from functools import partial
from threading import Event
from typing import List, Union

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image as ImageMsg
from ros_numpy.image import numpy_to_image, image_to_numpy

from .utils import draw_points
from .unet import ResNetUNet

size = 224
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--model', default='models/synth-e1+-lr1e-3-wd1e-4-fn.pth')
parser.add_argument('--max-crops', type=int, default=2)
args = parser.parse_args()

# TODO: potentially set max crops dynamically

device = torch.device(args.device)
model = ResNetUNet(2, pretrained=False).half().to(device)
model.load_state_dict(
    torch.load(args.model, map_location=lambda s, l: s)['model']
)


def infer(model, img: Image) -> np.ndarray:
    img = torchvision.transforms.ToTensor()(img)
    img = torchvision.transforms.Normalize(*imagenet_stats)(img)
    with torch.no_grad():
        result = model(img.half().unsqueeze(0).to(device)).squeeze(0)
    return result.detach().cpu().numpy()


def main():
    rospy.init_node('annotator')

    annotation_pubs = []
    annotated_img_pubs = []
    images = [None for _ in range(args.max_crops)]  # type: List[Union[None, ImageMsg]]
    new_data_event = Event()

    def cb(img_msg: ImageMsg, img_idx: int):
        images[img_idx] = img_msg
        new_data_event.set()

    for i in range(args.max_crops):
        annotated_img_pubs.append(
            rospy.Publisher('/servo/crop_{}/annotated'.format(i), ImageMsg, queue_size=1)
        )
        annotation_pubs.append(
            rospy.Publisher('/servo/crop_{}/points'.format(i), Float64MultiArray, queue_size=1)
        )
        rospy.Subscriber('/servo/crop_{}'.format(i), ImageMsg, partial(cb, img_idx=i), queue_size=1,
                         buff_size=size * size * 8 * 3 + 2 ** 8)

    while not rospy.is_shutdown():
        new_data_event.wait(1)
        new_data_event.clear()
        for i, annotation_pub, annotated_img_pub in zip(range(args.max_crops), annotation_pubs, annotated_img_pubs):
            # sequential inference in the main thread, interlaced between image sources
            # TODO: could be optimized by collating input from multiple queues
            #  and feeding the batch to the model instead
            img_msg = images[i]
            if img_msg is None:
                continue
            images[i] = None
            timestamp = img_msg.header.stamp.to_sec()
            img = image_to_numpy(img_msg)
            hms = infer(model, Image.fromarray(img))
            points = []
            for hm in hms:
                points.append(np.unravel_index(np.argmax(hm), hm.shape))
            points_flat = np.array(points)[:, ::-1].reshape(-1)
            points_flat = np.concatenate(([timestamp], points_flat))
            annotation_pub.publish(Float64MultiArray(data=points_flat.astype(np.float64)))
            for p, c in zip(points, 'br'):
                draw_points(img, [p], c=c)
            annotated_img_pub.publish(numpy_to_image(img, 'rgb8'))


if __name__ == '__main__':
    main()
