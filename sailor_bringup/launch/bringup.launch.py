import os
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():

    darknet_shared_dir = get_package_share_directory(
        "darknet_bringup")
    asus_xtion_shared_dir = get_package_share_directory(
        "asus_xtion")
    bringup_shared_dir = get_package_share_directory(
        "sailor_bringup")
    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_CONSOLE_STDOUT_LINE_BUFFERED", "1")

    #
    # NODES
    #
    features_extractor_node_cmd = Node(
        package="sailor",
        executable="features_extractor_node",
        name="features_extractor_node",
        output="screen",
        parameters=[{"target_frame": "base_link",
                     "maximum_detection_threshold": 0.2,
                     "histogram_bins_per_channel": 8,
                     "class_names": os.path.join(bringup_shared_dir, "config/darknet", "coco.names")}]
    )

    #
    # LAUNCHES
    #
    darknet_action_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(darknet_shared_dir, "launch",
                         "darknet.launch.py")),
        launch_arguments={
            "network_config": os.path.join(bringup_shared_dir, "config/darknet", "yolov3-tiny.cfg"),
            "weights": os.path.join(bringup_shared_dir, "config/darknet", "yolov3-tiny.weights"),
            "class_names": os.path.join(bringup_shared_dir, "config/darknet", "coco.names"),
            "threshold": "0.25",
            "nms_threshold": "0.50",
            "show_debug_image": "True",
            "input_image_topic": "/camera/rgb/image_raw"
        }.items()
    )

    asus_xtion_action_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(asus_xtion_shared_dir, "launch",
                         "asus_xtion.launch.py"))
    )

    ld = LaunchDescription()

    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(features_extractor_node_cmd)
    ld.add_action(darknet_action_cmd)
    ld.add_action(asus_xtion_action_cmd)

    return ld
