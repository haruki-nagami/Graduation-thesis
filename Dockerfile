FROM nvidia/cuda:12.4.0-base-ubuntu22.04

ARG python_version="3.10.5"

SHELL ["bash", "-c"]

ENV HOME /root

RUN apt-get update \
    && apt-get -y upgrade

RUN apt-get -y install curl git wget unzip dos2unix

# Install pyenv
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential libssl-dev libffi-dev libncurses5-dev zlib1g zlib1g-dev libreadline-dev libbz2-dev libsqlite3-dev liblzma-dev libopencv-dev
RUN curl https://pyenv.run | bash
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PATH:$PYENV_ROOT/bin
ENV PATH $PATH:/root/.pyenv/shims
RUN echo 'eval "$(pyenv init -)"' >> $HOME/.bashrc
RUN . ~/.bashrc

WORKDIR /workdir

# Initialize Python Environment
COPY ./requirements.txt ./requirements.txt
RUN pyenv install ${python_version} \
    && pyenv global ${python_version} \
    && pip install -r requirements.txt \
    && pip install jupyterlab \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

#ENTRYPOINT "bash"