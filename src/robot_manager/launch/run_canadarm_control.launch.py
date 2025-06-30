import os

from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    DeclareLaunchArgument,
    TimerAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    LogInfo,
)

from launch_ros.actions import Node, SetParameter
from launch.event_handlers.on_shutdown import OnShutdown
from ament_index_python.packages import get_package_share_directory

def launch_setup(context, *args, **kwargs):
    controller_node = Node(
        package="mppi_controller",
        executable="canadarm_controller_node",
        output='screen'
    )

    nodes_to_start =[
        controller_node
    ]
    return nodes_to_start


def generate_launch_description():
    declared_arguments = []

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
