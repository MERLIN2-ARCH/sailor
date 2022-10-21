import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration


from kant_dao.dao_factory import DaoFamilies


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
    # ARGS
    #
    matching_threshold = LaunchConfiguration("matching_threshold")
    matching_threshold_cmd = DeclareLaunchArgument(
        "matching_threshold",
        default_value="0.5",
        description="Matching threshold for anchoring process")

    dao_family = LaunchConfiguration("dao_family")
    dao_family_cmd = DeclareLaunchArgument(
        "dao_family",
        default_value=str(int(DaoFamilies.ROS2)),
        description="DAO family")

    mongo_uri = LaunchConfiguration("mongo_uri")
    mongo_uri_cmd = DeclareLaunchArgument(
        "mongo_uri",
        default_value="mongodb://localhost:27017/anchoring",
        description="MongoDB URI")

    #
    # NODES
    #
    features_extractor_node_cmd = Node(
        package="sailor",
        executable="features_extractor_node",
        name="features_extractor_node",
        output="screen",
        parameters=[{"target_frame": "base_link",
                     "maximum_detection_threshold": 0.7,
                     "histogram_bins_per_channel": 8,
                     "class_names": os.path.join(bringup_shared_dir, "config/darknet", "coco.names")}]
    )

    anchoring_node_cmd = Node(
        package="sailor",
        executable="anchoring_node",
        name="anchoring_node",
        output="screen",
        parameters=[{"matching_threshold": matching_threshold,
                     "mongo_uri": mongo_uri,
                     "dao_family": dao_family
                     }]
    )

    knowledge_base_node_cmd = Node(
        package="kant_knowledge_base",
        executable="knowledge_base_node.py",
        name="knowledge_base_node",
        condition=LaunchConfigurationEquals(
            "dao_family", str(int(DaoFamilies.ROS2)))
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

    ld.add_action(matching_threshold_cmd)
    ld.add_action(dao_family_cmd)
    ld.add_action(mongo_uri_cmd)

    ld.add_action(features_extractor_node_cmd)
    ld.add_action(anchoring_node_cmd)
    ld.add_action(knowledge_base_node_cmd)

    ld.add_action(darknet_action_cmd)
    ld.add_action(asus_xtion_action_cmd)

    return ld
