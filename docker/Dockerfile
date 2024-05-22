FROM nvcr.io/nvidia/tensorflow:24.03-tf2-py3

LABEL maintainer="alain.keller@ost.ch"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install ffmpeg libsm6 libxext6 xvfb -y

RUN python3 -m pip install --user --upgrade pip
RUN pip3 install --user \
	tensorboard \
	matplotlib \
	tqdm \
	ipdb \
	black \
	timm==0.5.4 \
  'gymnasium[mujoco]' \
  gymnasium==0.29.1 \
  tf_keras==2.15.1 \
  tensorflow-probability==0.23.0 \
  opencv-python
#	opencv-python
