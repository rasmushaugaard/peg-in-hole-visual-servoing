# peg-in-hole-visual-servoing

Visual servoing for peg-in-hole.  

The servoing consists of three nodes.
1) A crop node that captures and crops images from ros image topics.
2) An annotator node that processes the cropped images.
3) A client node that configures the crop node, subscribes to the annotations, and controls the robot(s).

The images can be captured and cropped by one computer, 
processed by a second computer (with a GPU) and 
the robot control can happen on a third computer.
If the computer that captures images is not the same as the computer that controls the robots,
then make sure, the computers clocks are synchronized, eg. with [chrony](https://chrony.tuxfamily.org/). 

Requires a trained model, see
[peg-in-hole-visual-servoing-model](https://github.com/RasmusHaugaard/peg-in-hole-visual-servoing-model).

#### install
``$ pip3 install -e .`` 

ROS is used for communication between the crop and client node.   
``$ catkin_make --directory ros``  
``$ source ros/devel/setup.bash``  


#### on the computer that is connected to the cameras
``python3 -m peg_in_hole_visual_servoing.crop``

#### on a computer with GPU
``python3 -m peg_in_hole_visual_servoing.annotator --model [model path]``

#### on the computer connected to the robots
```python
import json
from ur_control import Robot
import peg_in_hole_visual_servoing
from transform3d import SceneNode, SceneState

peg_robot = Robot.from_ip('192.168.1.123')
aux_robots = [Robot.from_ip('192.168.1.124')]
image_topics = '/camera_a/color/image_raw', '/camera_b/color/image_raw'

# build the scene structure
peg_robot_tcp, cams_robot_tcp, peg_robot_base, cams_robot_base, \
cam_a, cam_b, table = SceneNode.n(7)
table.adopt(
    peg_robot_base.adopt(peg_robot_tcp),
    cams_robot_base.adopt(cams_robot_tcp.adopt(cam_a, cam_b))
)
# insert necessary transforms from calibrations
state = SceneState()
state[peg_robot_base] = get_table_peg_base_calibration()
state[cams_robot_base] = get_table_cams_base_calibration()
state[cam_a] = get_tcp_cam_a_calibration()
state[cam_b] = get_tcp_cam_b_calibration()

Ks, dist_coeffs = get_camera_intrinsic_calibrations()

### Once, create a servo configuration:
# config_from_demonstration will let you move the robots in place for insertion
# and mark the holes in the images.
config = peg_in_hole_visual_servoing.config_from_demonstration(
        peg_robot=peg_robot, aux_robots=aux_robots,
        peg_tcp_node=peg_robot_tcp, aux_tcp_nodes=[cams_robot_tcp],
        scene_state=state,
        image_topics=image_topics, camera_nodes=[cam_a, cam_b],
        Ks=Ks, dist_coeffs=dist_coeffs
)

# the configuration is json serializable
json.dump(open('servo_config.json', 'w'), config)


### When servoing is needed
config = json.load(open('servo_config.json'))
peg_in_hole_visual_servoing.servo(
        peg_robot=peg_robot, aux_robots=aux_robots,
        peg_tcp_node=peg_robot_tcp, aux_tcp_nodes=[cams_robot_tcp],
        scene_state=state, camera_nodes=[cam_a, cam_b],
        servo_config=config, insertion_direction_tcp=(0, 0, 1)
)
```


