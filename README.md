# SAILOR (Symbolic AnchorIng from perceptuaL for rOs2-based Robots)

## Installation

```shell
# repos
$ cd ~/ros2_ws/src
$ git clone git@github.com:uleroboticsgroup/simple_node.git
$ git clone git@github.com:uleroboticsgroup/kant.git
$ git clone --recurse-submodules https://github.com/mgonzs13/openrobotics_darknet_ros.git
$ git clone https://github.com/mgonzs13/sailor.git

# kant dependencies
$ sudo pip3 install mongoengine dnspython

# Mongocxx
$ sudo apt install libmongoc-dev libmongoc-1.0-0 -y  # Ubuntu 20, mongoc 1.16.1
$ curl -OL https://github.com/mongodb/mongo-cxx-driver/archive/refs/tags/r3.4.2.tar.gz
$ tar -xzf r3.4.2.tar.gz
$ cd mongo-cxx-driver-r3.4.2/build
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBSONCXX_POLY_USE_BOOST=1
$ cmake --build .
$ sudo cmake --build . --target install
$ export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
$ echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
$ rm r3.4.2.tar.gz
$ rm -rf mongo-cxx-driver-r3.4.2

# MongoDB
$ wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
$ sudo apt-get install gnupg
$ wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
$ echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | sudo tee /etc/apt/$ sources.list.d/mongodb-org-4.4.list
$ sudo apt-get update
$ sudo apt-get install -y mongodb-org
$ sudo systemctl start mongod

# SAILOR dependencies
$ pip3 install -r sailor/requirements.txt

# rosdep
$ cd ~/ros2_ws
$ rosdep install --from-paths src -r -y

# colcon
$ colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DDARKNET_OPENCV=Off --packages-select darknet_vendor
$ colcon build
```
