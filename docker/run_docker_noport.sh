xhost +local:
docker run --rm -it \
  --gpus all \
  --cpus=6 \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $PWD:/home/drl \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
  -p 8083:6006 \
  drl:0.1
