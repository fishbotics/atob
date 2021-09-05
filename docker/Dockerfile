FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04

# Set variables
ARG USERNAME
ARG USER_ID

# Set environment variables
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

# Set environment variables necessary to execute Click CLI commands in the docker image
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


# See: https://github.com/phusion/baseimage-docker/issues/58
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install some basic things
RUN apt update && apt install -y --no-install-recommends \
  tzdata

RUN ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
RUN echo "America/Los_Angeles" > /etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata

# Add sudo command
RUN apt update && apt-get install sudo

# Set up sudo user
# RUN adduser --disabled-password --uid $USER_ID --gecos '' $USERNAME
RUN adduser --disabled-password --gecos '' $USERNAME
RUN adduser $USERNAME sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $USERNAME
WORKDIR /home/$USERNAME

RUN sudo apt update && sudo apt install -y --no-install-recommends \
  tmux \
  wget \
  htop \
  neovim \
  lsb-release \
  python3-neovim

# This is all stuff needed for OMPL, taken from https://ompl.kavrakilab.org/core/install-ompl-ubuntu.sh
RUN sudo apt update && sudo apt install -y --no-install-recommends \
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
  python3-dev \
  python3-pip \
  libboost-python-dev \
  libboost-numpy-dev \
  python3-numpy \
# Python 19.10 or greater, otherwise you need special commands
  castxml \
  pypy3 \
  silversearcher-ag \
  libpng16-16 \
  git


# Back to normal
RUN sudo apt-get clean && sudo rm -rf /var/lib/apt

RUN git clone https://github.com/jbeder/yaml-cpp \
  && cd yaml-cpp \
  && git checkout yaml-cpp-0.6.3 \
  && mkdir build \
  && cd build \
  && cmake -DYAML_BUILD_SHARED_LIBS=ON -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF .. \
  && make -j \
  && sudo make install


RUN python3 -m pip install -vU https://github.com/CastXML/pygccxml/archive/develop.zip pyplusplus

RUN wget -O - https://github.com/ompl/ompl/archive/refs/tags/1.5.2.tar.gz | tar zxf -
RUN mkdir -p ompl-1.5.2/build/Release
WORKDIR ompl-1.5.2/build/Release
RUN export CXX=g++
RUN export MAKEFLAGS="-j `nproc`"
RUN cmake ../.. -DPYTHON_EXEC=/usr/bin/python3
RUN make update_bindings
# Can't build with -j flag or else the build process for the bindings runs out of memory
RUN make
RUN sudo make install
WORKDIR /

RUN sudo update-alternatives --install /usr/bin/vi vi /usr/bin/nvim 60 \
  && sudo update-alternatives --config vi \
  && sudo update-alternatives --install /usr/bin/vim vim /usr/bin/nvim 60 \
  && sudo update-alternatives --config vim \
  && sudo update-alternatives --install /usr/bin/editor editor /usr/bin/nvim 60 \
  && sudo update-alternatives --config editor

RUN python3 -m pip install -U numpy ipython pybullet pyquaternion h5py termcolor
RUN python3 -m pip install git+https://github.com/yijiangh/ikfast_pybind/

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)"
RUN echo "alias ll='ls -lhaFtr'" >> ~/.zshrc
RUN echo "alias ta='tmux attach-session -dt'" >> ~/.zshrc
RUN echo "alias tn='tmux new -s'" >> ~/.zshrc
RUN echo "alias sl='du -sh * | sort -h'" >> ~/.zshrc
RUN echo "set-option -g default-shell /bin/zsh" >> ~/.tmux.conf
ENV PYTHONPATH=/home/atob:$PYTHONPATH
