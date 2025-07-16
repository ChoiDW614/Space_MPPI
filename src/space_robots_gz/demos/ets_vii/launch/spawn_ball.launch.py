import os
import xacro

from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    simulation_models_path = get_package_share_directory('simulation')
    sphere_urdf_path = os.path.join(simulation_models_path, 'models', 'sphere', 'spawn_spheres.urdf.xacro')

    doc = xacro.process_file(sphere_urdf_path)
    robot_description = {'robot_description': doc.toxml()}

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen'
    )

    spawn_spheres_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'spawned_spheres',
            '-allow_renaming', 'true',
            '-x', '0.0', '-y', '0.0', '-z', '0.0'
        ],
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher_node,
        spawn_spheres_node
    ])

