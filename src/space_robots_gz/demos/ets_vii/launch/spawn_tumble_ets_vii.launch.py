from http.server import executable
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.substitutions import TextSubstitution, PathJoinSubstitution, LaunchConfiguration, Command
from launch_ros.actions import Node
from launch.actions import TimerAction
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit, OnExecutionComplete
import os
from os import environ

from ament_index_python.packages import get_package_share_directory

import xacro


def generate_launch_description():
    simulation_models_path = get_package_share_directory('simulation')
    ets_vii_urdf_model_path = os.path.join(simulation_models_path, 'models', 'ets_vii', 'urdf', 'ets_vii.urdf.xacro')

    ets_vii_doc = xacro.process_file(ets_vii_urdf_model_path)
    ets_vii_robot_description = {'robot_description': ets_vii_doc.toxml()}

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ets_vii_robot_state_publisher',
        namespace='ets_vii',
        output='screen',
        parameters=[ets_vii_robot_description]
    )

    spawn = Node(
        package='ros_ign_gazebo', executable='create',
        namespace='ets_vii',
        arguments=[
            '-name', 'ets_vii',
            '-topic', ets_vii_robot_description,
            # '-x', '-2.1649', '-y', '4.4368', '-z', '5.3509',
            # '-R', '-1.43', '-P', '0.16', '-Y', '1.71',
            '-x', '0.0', '-y', '0.0', '-z', '8.0',
            '-R', '0.0', '-P', '0.0', '-Y', '0.0',
        ],
        output='screen'
    )

    pose_publisher = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/model/ets_vii/pose@geometry_msgs/msg/TransformStamped@ignition.msgs.Pose'],
    )

    return LaunchDescription([
        robot_state_publisher,
        spawn,
        pose_publisher,
    ])
