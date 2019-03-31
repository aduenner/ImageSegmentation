ARG cuda_version=9.0
ARG cudnn_version=7
#FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel
FROM tensorflow/tensorflow:1.12.0-gpu-py3
# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      libglib2.0-dev \
      sudo \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install \
	sklearn-pandas \
      	cntk-gpu \
        bcolz \
        h5py \
        matplotlib \
        mkl \
        nose \
        notebook \
        Pillow \
        pandas \
        pydot \
        pm4pygpu \
        PyYAML \
        scikit-learn \
        six \
        mkdocs \
        opencv-python \
        jupyterlab \
        tqdm \
        scikit-image \
        multiprocess \
        nibabel \
        nipype \
        SimpleITK \
      && \
    git clone git://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
    pip install git+git://github.com/keras-team/keras.git

ENV NB_USER nbuser
ENV NB_UID 1000
RUN adduser --disabled-password --gecos '' nbuser
RUN adduser nbuser sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH='/src/:$PYTHONPATH'
USER 0:0
WORKDIR /

