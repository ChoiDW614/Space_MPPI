from http.server import executable
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, IncludeLaunchDescription
from launch.substitutions import TextSubstitution, PathJoinSubstitution, LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit, OnExecutionComplete
import os
from os import environ

from ament_index_python.packages import get_package_share_directory

import xacro


def generate_launch_description():
    # ld = LaunchDescription()
    franka_demos_path = get_package_share_directory('franka')
    simulation_models_path = get_package_share_directory('simulation')

    env = {'IGN_GAZEBO_SYSTEM_PLUGIN_PATH':
           ':'.join([environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', default=''),
                     environ.get('LD_LIBRARY_PATH', default='')]),
           'IGN_GAZEBO_RESOURCE_PATH':
           ':'.join([environ.get('IGN_GAZEBO_RESOURCE_PATH', default=''), franka_demos_path])}


    urdf_model_path = os.path.join(simulation_models_path, 'models', 'franka', 'urdf', 'franka.urdf.xacro')
    leo_model = os.path.join(franka_demos_path, 'worlds', 'empty.world')

    doc = xacro.process_file(urdf_model_path)
    robot_description = {'robot_description': doc.toxml()}

    start_world = ExecuteProcess(
        cmd=['ign gazebo', leo_model, '-r'],
        output='screen',
        additional_env=env,
        shell=True
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    spawn = Node(
        package='ros_ign_gazebo', executable='create',
        arguments=[
            '-name', 'franka',
            '-topic', robot_description,
            '-x', '0.0', '-y', '0.0', '-z', '0.0',
            '-R', '0.0', '-P', '0.0', '-Y', '0.0',
        ],
        output='screen'
    )

    pose_publisher = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/world/default/pose/info@geometry_msgs/msg/PoseArray@ignition.msgs.Pose_V'
        ],
        shell=False
    )

    # Control
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'franka_joint_state_broadcaster'],
        output='screen'
    )

    load_franka_joint_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'franka_joint_controller'],
        output='screen'
    )

    franka_wrapper_spawn = Node(
        package="mppi_controller",
        executable="franka_controller_node",
        output='screen'
    )

    return LaunchDescription([
        start_world,
        robot_state_publisher,
        pose_publisher,
        spawn,

        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_franka_joint_controller],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_franka_joint_controller,
                on_exit=[franka_wrapper_spawn],
            )
        ),
    ])
