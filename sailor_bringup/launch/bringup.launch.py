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

    yolo_shared_dir = get_package_share_directory(
        "yolov8_bringup")
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

    torch_device = LaunchConfiguration("torch_device")
    torch_device_cmd = DeclareLaunchArgument(
        "torch_device",
        default_value="cuda:0",
        description="The device used in Pytorch (cuda, cpu)")

    weights_path = LaunchConfiguration("weights_path")
    weights_path_cmd = DeclareLaunchArgument(
        "weights_path",
        default_value=os.path.join(
            bringup_shared_dir, "weights/mix/dl_model.pt"
        ),
        description="Path to the weights")

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
    percept_generator_node_cmd = Node(
        package="sailor",
        executable="percept_generator_node",
        name="percept_generator_node",
        output="screen",
        parameters=[{"target_frame": "base_link",
                     "detection_score_threshold": 0.5,
                     "torch_device": torch_device
                     }]
    )

    anchoring_node_cmd = Node(
        package="sailor",
        executable="anchoring_node",
        name="anchoring_node",
        output="screen",
        parameters=[{"matching_threshold": matching_threshold,
                     "torch_device": torch_device,
                     "weights_path": weights_path,
                     "mongo_uri": mongo_uri,
                     "dao_family": dao_family,
                     }]
    )

    knowledge_base_node_cmd = Node(
        package="kant_knowledge_base",
        executable="knowledge_base_node.py",
        name="knowledge_base_node",
        condition=LaunchConfigurationEquals(
            "dao_family", str(int(DaoFamilies.ROS2)))
    )

    rviz_cmd = Node(
        name="rviz",
        package="rviz2",
        executable="rviz2",
        arguments=["-d", os.path.join(
            bringup_shared_dir,
            "rviz",
            "default.rviz")]
    )

    #
    # LAUNCHES
    #
    yolo_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(yolo_shared_dir, "launch",
                         "yolov8.launch.py")),
        launch_arguments={
            "model": "yolov8x.pt",
            "device": torch_device,
            "enable": "True",
            "threshold": "0.7",
            "input_image_topic": "/camera/rgb/image_raw",
            "namespace": "yolo"
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
    ld.add_action(torch_device_cmd)
    ld.add_action(weights_path_cmd)
    ld.add_action(dao_family_cmd)
    ld.add_action(mongo_uri_cmd)

    ld.add_action(percept_generator_node_cmd)
    ld.add_action(anchoring_node_cmd)
    ld.add_action(knowledge_base_node_cmd)
    ld.add_action(rviz_cmd)

    ld.add_action(yolo_cmd)
    ld.add_action(asus_xtion_action_cmd)

    return ld
