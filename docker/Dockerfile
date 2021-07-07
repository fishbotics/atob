FROM ubuntu:20.04

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
  tmux \
  wget \
  htop \
  neovim \
  lsb-release \
  python3-neovim \
  && apt-get clean && rm -rf /var/lib/apt
RUN wget --no-check-certificate https://ompl.kavrakilab.org/core/install-ompl-ubuntu.sh
RUN chmod u+x install-ompl-ubuntu.sh
RUN ./install-ompl-ubuntu.sh --python

RUN update-alternatives --install /usr/bin/vi vi /usr/bin/nvim 60 \
  && update-alternatives --config vi \
  && update-alternatives --install /usr/bin/vim vim /usr/bin/nvim 60 \
  && update-alternatives --config vim \
  && update-alternatives --install /usr/bin/editor editor /usr/bin/nvim 60 \
  && update-alternatives --config editor

RUN pip3 install ipython pybullet pyquaternion

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)"
RUN echo "alias ll='ls -lhaFtr'" >> ~/.zshrc
RUN echo "alias ta='tmux attach-session -dt'" >> ~/.zshrc
RUN echo "alias tn='tmux new -s'" >> ~/.zshrc
RUN echo "alias sl='du -sh * | sort -h'" >> ~/.zshrc
RUN echo "set-option -g default-shell /bin/zsh" >> ~/.tmux.conf
