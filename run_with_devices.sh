#!/bin/bash

xhost +local:root

docker run --rm \
  -e SDL_AUDIODRIVER=dummy \
  -e QT_QPA_PLATFORM=xcb \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/snd \
  --device /dev/video0 \
  --device /dev/video1 \
  -p 8000:8000 \
  acv_fitness_project:latest

# Make it executable: chmod +x run_with_devices.sh
# Then just run: bash run_with_devices.sh

# Removed
# -e SDL_AUDIODRIVER=pulse \