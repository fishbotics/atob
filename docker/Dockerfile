FROM ubuntu:20.04

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
  tmux \
  wget \
  htop \
  neovim \
  lsb-release \
  python3-neovim \
  && apt-get clean && rm -rf /var/lib/apt

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
  g++ \
  make \
  cmake \
  pkg-config \
  libboost-serialization-dev \
  libboost-filesystem-dev \
  libboost-system-dev \
  libboost-program-options-dev \
  libboost-test-dev \
  libeigen3-dev \
  libode-dev \
  libyaml-cpp-dev \
  python3-dev \
  python3-pip \
  libboost-python-dev \
  libboost-numpy-dev \
  python3-numpy \
# Python 19.10 or greater, otherwise you need special commands
  castxml \
  pypy3 \
# Back to normal
  && apt-get clean && rm -rf /var/lib/apt

# RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.bz2

RUN python3 -m pip install -vU https://github.com/CastXML/pygccxml/archive/develop.zip pyplusplus

RUN wget -O - https://github.com/ompl/ompl/archive/refs/tags/1.5.2.tar.gz | tar zxf -
RUN mkdir -p ompl-1.5.2/build/Release
WORKDIR ompl-1.5.2/build/Release
RUN export CXX=g++
RUN export MAKEFLAGS="-j `nproc`"
RUN cmake ../.. -DPYTHON_EXEC=/usr/bin/python3
RUN make update_bindings
RUN make
RUN make install
WORKDIR /



# These commands don't need to be run because now the commands are part of the docker file,
# but they are left in for posterity in case it stops working in the future
# RUN wget --no-check-certificate https://ompl.kavrakilab.org/core/install-ompl-ubuntu.sh
# RUN chmod u+x install-ompl-ubuntu.sh
# RUN ./install-ompl-ubuntu.sh --python

RUN update-alternatives --install /usr/bin/vi vi /usr/bin/nvim 60 \
  && update-alternatives --config vi \
  && update-alternatives --install /usr/bin/vim vim /usr/bin/nvim 60 \
  && update-alternatives --config vim \
  && update-alternatives --install /usr/bin/editor editor /usr/bin/nvim 60 \
  && update-alternatives --config editor

RUN python3 -m pip install ipython pybullet pyquaternion

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)"
RUN echo "alias ll='ls -lhaFtr'" >> ~/.zshrc
RUN echo "alias ta='tmux attach-session -dt'" >> ~/.zshrc
RUN echo "alias tn='tmux new -s'" >> ~/.zshrc
RUN echo "alias sl='du -sh * | sort -h'" >> ~/.zshrc
RUN echo "set-option -g default-shell /bin/zsh" >> ~/.tmux.conf
ENV PYTHONPATH=/atob:$PYTHONPATH
