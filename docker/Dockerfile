# based on https://github.com/runpod/containers/blob/main/torch/Dockerfile
# MIT License, Copyright (c) 2022 Run-Pod

ARG BASE_IMAGE=nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
ARG REPO
ARG TOKEN

FROM ${BASE_IMAGE} as dev-base

WORKDIR /

RUN mkdir /workspace

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash
RUN apt-get update --yes && \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends\
    git\
    wget\
    curl\
    git\
    bash\
    software-properties-common\
    openssh-server
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install --yes --no-install-recommends\
    python3.10\
    python3.10-distutils\
    python3.10-venv\
    python3-pip && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py
RUN pip install -U jupyterlab ipywidgets jupyter-archive
RUN jupyter nbextension enable --py widgetsnbextension
RUN python -m venv /workspace/.venv
RUN /workspace/.venv/bin/pip install poetry
RUN /workspace/.venv/bin/poetry config virtualenvs.in-project true
# RUN git clone https://${TOKEN}@github.com/{$REPO}.git /workspace
# RUN /workspace/.venv/bin/poetry install


RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

ADD start.sh /

RUN chmod +x /start.sh

WORKDIR /workspace

CMD [ "/start.sh" ]
