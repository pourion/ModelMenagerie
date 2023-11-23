# for contents see https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
FROM nvcr.io/nvidia/pytorch:23.05-py3

ARG GITHUB_BRANCH=main
ARG GITHUB_REPO=github.com:JAX-DIPS/JAX-DIPS.git
ARG DIPS_HOME=/opt/DIPS


RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y git vim sudo gpustat libopenexr-dev python3-pybind11 libx11-6


COPY ./requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt

RUN cd /opt \
    && git clone https://github.com/paulo-herrera/PyEVTK.git \
    && cd PyEVTK/ \
    && python3 setup.py install

ENV PYTHONPATH="/opt/PyEVTK"


RUN cd /opt \
    && git clone https://github.com/tinyobjloader/tinyobjloader \
    && cd tinyobjloader \
    && python -m pip install .

# RUN mkdir -p /workspace/third_party \
#     && cd /workspace/third_party \
#     && git clone https://github.com/nv-tlabs/nglod.git \
#     && cd /workspace/third_party/nglod/sdf-net/lib/extensions \
#     && bash build_ext.sh

# Clone using SSH (add the domain as known before)
# RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan ${GITHUB_REPO%%[:/]*} >> ~/.ssh/known_hosts
# RUN --mount=type=ssh git clone --branch ${GITHUB_BRANCH} ssh://git@${GITHUB_REPO} ${DIPS_HOME}