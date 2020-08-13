#!/usr/bin/env python3
from setuptools import setup

setup(
    name='peg_in_hole_visual_servoing',
    version='0.0.0',
    install_requires=[
        'numpy',
        'pillow',
        'typing',
        'opencv-python',
        'transform3d',
        'ur_control',
        'torch',
        'torchvision'
    ]
)
