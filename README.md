# SAILOR (Symbolic AnchorIng from perceptuaL for rOs2-based Robots)

## Installation

```shell
$ cd ~/ros2_ws/src
$ git clone https://github.com/mgonzs13/sailor.git
$ pip3 install -r sailor/requirements.txt
$ cd ~/ros2_ws
$ rosdep install --from-paths src -r -y
$ colcon build
```
