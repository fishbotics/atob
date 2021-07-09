# atob
This library is meant to be an easy-to-use planning framework written in Python. The intention of this project is to create a standard, simple, fast, and flexible planner with a stable Python API. While working toward this, the guts that make this possible may change significantly. Currently, it is written to use Pybullet for collision checking and OMPL for planning.

The easiest way to get started is to build the code using the included docker file. This will install the necessary dependencies. I'd like it if this could be installed via `pip` or `conda` someday, but that requires quite a few changes for OMPL, which is currently very not `pip` friendly.

First, clone the repo. You can clone it wherever you want, but I'll use `$HOME` in the documentation.
```
cd $HOME
git clone https://github.com/fishbotics/atob
cd $HOME/atob
```
Now, you have to build the docker container. This requires you have docker installed. Building will take a while. If a step fails, it may be a network problem, so try rerunning it. If it still fails, post an issue on this Github.
```
docker build \                                                                                                                                                                                                                                   ─╯
  --build-arg "USERNAME=$USER" \
  --tag atob \
  --network=host \
  --file docker/Dockerfile \
  .
```
After it builds, run the container as follows (change the path to this repo to
match where you cloned it on your local system
```
docker run \                                                                                                                                                                                                                                     ─╯
  --interactive \
  --tty \
  --rm \
  --user $USER \
  --privileged \
  --gpus all \
  --network host \
  --env DISPLAY=unix$DISPLAY \
  --env XAUTHORITY \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env LC_ALL=C.UTF-8 \
  --env LANG=C.UTF-8 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --volume /dev/input:/dev/input \
  --volume $HOME/atob/:/atob \
  --workdir /atob \
  atob \
  /bin/zsh
```
Once inside the docker container, you can pull up a Python shell with `ipython` or `python3` and then run:
```
from atob.planner import Planner
from atob.geometry import Cuboid

# You can give a different URDF, although the Franka details are hardcoded for now, so you might as well use this URDF
planner = Planner('/atob/urdf/franka_panda/panda.urdf')

# Cuboids are defined from the center, dimensions, and then the [w, x, y, z] quaternion
obstacle = Cuboid([0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [1, 0, 0, 0])

# You can include as many cuboids as you want
planner.create_scene([obstacle])

path = planner.plan(start=[0, 0, 0, -1, 0, 0, 0], goal=[1, 1, 1, -1, 1, 1, 1])
print(path)
```
