FROM huggingface/transformers-pytorch-gpu:latest

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y git vim sudo gpustat libopenexr-dev python3-pybind11 libx11-6

COPY ./requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt