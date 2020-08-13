cmake_minimum_required(VERSION 2.8.3)
project(peg_in_hole_visual_servoing_api)

find_package(catkin REQUIRED COMPONENTS genmsg message_generation std_msgs sensor_msgs)

add_service_files(
    FILES
    SetString.srv
)

generate_messages(
    DEPENDENCIES
    std_msgs
)

catkin_package(
    CATKIN_DEPENDS message_runtime std_msgs sensor_msgs
)