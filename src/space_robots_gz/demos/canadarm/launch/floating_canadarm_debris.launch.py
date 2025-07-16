from http.server import executable
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, IncludeLaunchDescription, TimerAction
from launch.substitutions import TextSubstitution, PathJoinSubstitution, LaunchConfiguration, Command
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
import os
from os import environ

from ament_index_python.packages import get_package_share_directory

import xacro


def generate_launch_description():
    # ld = LaunchDescription()
    canadarm_demos_path = get_package_share_directory('canadarm')
    simulation_models_path = get_package_share_directory('simulation')

    env = {'IGN_GAZEBO_SYSTEM_PLUGIN_PATH':
           ':'.join([environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', default=''),
                     environ.get('LD_LIBRARY_PATH', default='')]),
           'IGN_GAZEBO_RESOURCE_PATH':
           ':'.join([environ.get('IGN_GAZEBO_RESOURCE_PATH', default=''), canadarm_demos_path])}

    urdf_model_path = os.path.join(simulation_models_path, 'models', 'canadarm', 'urdf', 'floating_canadarm.urdf.xacro')
    leo_model = os.path.join(canadarm_demos_path, 'worlds', 'simple_wo_iss.world')

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
            '-name', 'canadarm',
            '-topic', robot_description,
            '-x', '0.0', '-y', '0.0', '-z', '-2.5',
            '-R', '0.0', '-P', '0.0', '-Y', '0.0',
        ],
        output='screen'
    )

    pose_publisher = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/model/canadarm/pose@geometry_msgs/msg/TransformStamped@ignition.msgs.Pose'],
    )

    clock = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=['/world/default/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen'
    )

    # Control
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'canadarm_joint_state_broadcaster'],
        output='screen'
    )

    load_canadarm_joint_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'floating_canadarm_joint_controller'],
        output='screen'
    )

    canadarm_wrapper_spawn = Node(
        package="mppi_controller",
        executable="canadarm_controller_node",
        output='screen',
        parameters=[{'config_file': 'reach_floating.yaml'}]
    )

    # Target spawn
    ets_vii_target_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('ets_vii'),
                'launch/spawn_tumble_ets_vii.launch.py')),
        launch_arguments=[]
    )

    ets_vii_tumbling = ExecuteProcess(
        cmd=['ign', 'topic', '-t', '/world/default/wrench', '-m', 'ignition.msgs.EntityWrench',
            '-p', '\"entity:', '{name:', '\'ets_vii\',', 'type:', 'MODEL},', 'wrench:', '{force:', '{x:50000,', 'y:50000},',
            'torque:', '{x:50000,', 'y:50000,', 'z:50000}}\"'],
        output='screen',
        shell=True
    )

    delay_ets_vii_tumbling = TimerAction(
        period=6.0,
        actions=[ets_vii_tumbling]
    )

    # Debris spawn
    sphere_sdf_path = os.path.join(simulation_models_path, 'models', 'sphere', 'sphere.sdf')
    debris_list = [
        {'name': 'ball1', 'x': '-3.0', 'y': '-2.0', 'z': '6.0'},
        {'name': 'ball2', 'x': '2.0',  'y': '1.0',  'z': '5.0'},
    ]

    debris_spawn_node = []
    for debris in debris_list:
        debris_spawn_node.append(
            Node(package='ros_ign_gazebo',
                executable='create',
                name=f"spawn_{debris['name']}",
                arguments=[
                    '-file', sphere_sdf_path,
                    '-name', debris['name'],
                    '-x', debris['x'], '-y', debris['y'], '-z', debris['z']
                ],
                output='screen'
            )
        )
    
    debris_pose_publisher = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/world/default/dynamic_pose/info@geometry_msgs/msg/PoseArray@ignition.msgs.Pose_V'],
    )

    return LaunchDescription([
        start_world,
        robot_state_publisher,
        pose_publisher,
        clock,
        spawn,
        ets_vii_target_spawn,
        delay_ets_vii_tumbling,
        *debris_spawn_node,
        debris_pose_publisher,

        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_canadarm_joint_controller],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_canadarm_joint_controller,
                on_exit=[canadarm_wrapper_spawn],
            )
        ),
    ])
