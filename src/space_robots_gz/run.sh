#!/usr/bin/env bash

# Runs a docker container with the image created by build.bash
# Requires:
#   docker
#   an X server

IMG_NAME=openrobotics/space_robots_demo

# Replace `/` with `_` to comply with docker container naming
# And append `_runtime`
CONTAINER_NAME="$(tr '/' '_' <<< "$IMG_NAME")"

# Start the container
docker run --rm -it --name $CONTAINER_NAME --runtime=nvidia --gpus all --network host \
    -v ${HOME}/space_ws/.ros/cyclonedds_local.xml:/home/spaceros-user/.ros/cyclonedds_local.xml \
    -v ${HOME}/space_ws/.vscode:/home/spaceros-user/demos_ws/.vscode \
    -v ${HOME}/space_ws/src/mppi_controller:/home/spaceros-user/demos_ws/src/mppi_controller \
    -v ${HOME}/space_ws/src/robot_manager:/home/spaceros-user/demos_ws/src/robot_manager \
    -v ${HOME}/space_ws/src/space_robots_gz/demos:/home/spaceros-user/demos_ws/src/demos \
    -v ${HOME}/space_ws/src/space_robots_gz/simulation:/home/spaceros-user/demos_ws/src/simulation \
    -e DISPLAY -e TERM -e QT_X11_NO_MITSHM=1 $IMG_NAME
