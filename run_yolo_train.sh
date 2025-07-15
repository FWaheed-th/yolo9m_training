#!/bin/bash

CONTAINER_NAME=ai4green-yolo-training

# Build the Docker image if not already built
docker build -f docker/Dockerfile -t ai4green_yolov9-training .

# Run the container in detached mode and mount volumes
docker run -dit \
  --gpus all \
  --name $CONTAINER_NAME \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/weights:/app/weights \
  -v ${PWD}/runs:/app/runs \
  -v ${PWD}/evaluation:/app/evaluation \
  ai4green_yolov9-training bash

# Start training inside the container
docker exec -it $CONTAINER_NAME bash -c "cd /app && python notebooks/train.py --mode gpu --epochs 300 --batch 16 && bash"
