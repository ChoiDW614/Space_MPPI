from http.server import executable
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, IncludeLaunchDescription, TimerAction
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
    canadarm_demos_path = get_package_share_directory('canadarm')
    simulation_models_path = get_package_share_directory('simulation')

    env = {'IGN_GAZEBO_SYSTEM_PLUGIN_PATH':
           ':'.join([environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', default=''),
                     environ.get('LD_LIBRARY_PATH', default='')]),
           'IGN_GAZEBO_RESOURCE_PATH':
           ':'.join([environ.get('IGN_GAZEBO_RESOURCE_PATH', default=''), canadarm_demos_path])}


    urdf_model_path = os.path.join(simulation_models_path, 'models', 'canadarm', 'urdf', 'floating_canadarm_camera.urdf.xacro')
    leo_model = os.path.join(canadarm_demos_path, 'worlds', 'simple_wo_iss_display.world')

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

    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/SSRMS_camera/ee/image_raw', '/SSRMS_camera/base/image_raw'],
        output='screen'
    )

    camera_info_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/SSRMS_camera/ee/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo',
                   '/SSRMS_camera/base/camera_info@sensor_msgs/msg/CameraInfo@ignition.msgs.CameraInfo'],
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


    return LaunchDescription([
        start_world,
        robot_state_publisher,
        pose_publisher,
        image_bridge,
        camera_info_bridge,
        spawn,
        ets_vii_target_spawn,
        delay_ets_vii_tumbling,

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
    ])
