

ROS2HOME=~/ros2_ws/
ROS2HOME_SRC=~/ros2_ws/src

dowload_sailor_dependencies(){
    cd $ROS2HOME_SRC

    git clone git@github.com:uleroboticsgroup/simple_node.git
    git clone git@github.com:uleroboticsgroup/kant.git
    git clone --recurse-submodules https://github.com/mgonzs13/ros2_asus_xtion
    git clone --recurse-submodules https://github.com/mgonzs13/openrobotics_darknet_ros.git
    git clone git@github.com:mgonzs13/sailor.git
}

dowload_sailor(){
    cd $ROS2HOME_SRC
    git clone git@github.com:mgonzs13/sailor.git
    # SAILOR dependencies
    cd src/sailor 
    pip3 install -r sailor/requirements.txt
}

install_kant_dependencies(){
    # kant dependencies
    sudo apt install python3-pip git python3-rosdep2
    sudo pip3 install mongoengine dnspython
}

# Ubuntu 20, mongoc 1.16.1
install_mongocxx_dependencies_ubuntu20(){
    cd $HOME
    # Mongocxx
    sudo apt install libmongoc-dev libmongoc-1.0-0 -y  
    curl -OL https://github.com/mongodb/mongo-cxx-driver/archive/refs/tags/r3.4.2.tar.gz
    tar -xzf r3.4.2.tar.gz
    cd mongo-cxx-driver-r3.4.2/build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBSONCXX_POLY_USE_BOOST=1
    cmake --build .
    sudo cmake --build . --target install
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    rm r3.4.2.tar.gz
    rm -rf mongo-cxx-driver-r3.4.2
}

# Ubuntu 22, mongoc 1.16.1
install_mongocxx_dependencies_ubuntu22(){
    cd $HOME
    # Mongocxx
    sudo apt install libmongoc-dev libmongoc-1.0-0 -y  
    curl -OL https://github.com/mongodb/mongo-cxx-driver/releases/download/r3.6.7/mongo-cxx-driver-r3.6.7.tar.gz
    tar -xzf mongo-cxx-driver-r3.6.7.tar.gz
    cd mongo-cxx-driver-r3.6.7/build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBSONCXX_POLY_USE_BOOST=1
    cmake --build .
    sudo cmake --build . --target install
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    cd 
    rm mongo-cxx-driver-r3.6.7.tar.gz
    rm -rf mongo-cxx-driver-r3.6.7
}

install_mongodb_20(){
    
    # MongoDB
    wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
    sudo apt-get install gnupg
    wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | sudo tee /etc/apt/$ sources.list.d/mongodb-org-4.4.list
    sudo apt-get update
    sudo apt-get install -y mongodb-org
    sudo systemctl start mongodb
}

install_mongodb_22(){
    # Ubuntu 22 trick
    # https://askubuntu.com/questions/1403619/mongodb-install-fails-on-ubuntu-22-04-depends-on-libssl1-1-but-it-is-not-insta
    wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb
    dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb
    rm libssl1.1_1.1.1f-1ubuntu2_amd64.deb

    
    # MongoDB
    # https://wiki.crowncloud.net/?How_to_Install_Latest_MongoDB_on_Ubuntu_22_04
    sudo apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common
    wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo apt-key add -
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
    sudo apt-get update
    sudo apt-get install -y mongodb-org
    sudo systemctl start mongod.service
    sudo systemctl status mongod.service

}

launch_rosdep(){
    # rosdep
    cd $ROS2HOME
    rosdep install --from-paths src -r -y
}

launch_colcon(){
    # colcon
    cd $ROS2HOME
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DDARKNET_OPENCV=Off --packages-select darknet_vendor
    colcon build
}


#install_kant_dependencies
install_mongocxx_dependencies_ubuntu22
install_mongodb_22
#dowload_sailor_dependencies
#dowload_sailor
#sudo apt-get install python3-rosdep2
#launch_rosdep
#launch_colcon
