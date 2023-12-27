FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    lsb-release \
    gnupg2 \
    protobuf-compiler \
    libeigen3-dev \
    libopencv-dev 
    # python-rosdep \
    # python-rosinstall \
    # python-rosinstall-generator \
    # python-wstool \
    # build-essential

# Download and run ubuntu_sim_common_deps.sh
RUN apt-get update && apt-get install -y wget && \
    wget https://raw.githubusercontent.com/PX4/Devguide/master/build_scripts/ubuntu_sim_common_deps.sh -O - | bash

# Add ROS repository and keys
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt-get update

# RUN pip install python-catkin-pkg-modules
RUN apt-get install -y ros-melodic-desktop-full

# Install necessary tools and initialize rosdep
RUN apt-get update && \
    apt-get install -y lsb-release python-rosdep && \
    rosdep init && \
    rosdep update

# Setup environment variables
# Note: For Docker, it's more common to set environment variables or source
# scripts at runtime, not in the Dockerfile, as the latter only affects the build process
ENV ROS_SOURCE="/opt/ros/melodic/setup.bash"

# The following command checks if the setup.bash line exists in .bashrc,
# and if not, adds it. However, this is typically not needed in Docker containers.
RUN if grep -Fxq "$ROS_SOURCE" ~/.bashrc; then \
        echo "ROS setup.bash already in .bashrc"; \
    else \
        echo "$ROS_SOURCE" >> ~/.bashrc; \
    fi

# Source the ROS setup script
# Note: This will only affect the layers during the image build. 
# For runtime sourcing, consider using an entrypoint script.
RUN /bin/bash -c 'source /opt/ros/melodic/setup.bash'

# Install MAVROS dependencies
RUN apt-get install -y python-catkin-tools python-rosinstall-generator

# Create catkin workspace
RUN apt-get install -y python-wstool
RUN mkdir -p ~/catkin_ws/src && \
    cd ~/catkin_ws && \
    catkin init && \
    catkin config --merge-devel && \
    wstool init src

# Install MAVLink
RUN cd ~/catkin_ws && \
    rosinstall_generator --rosdistro kinetic mavlink | tee /tmp/mavros.rosinstall

# Build MAVROS
RUN /bin/bash -c 'source /opt/ros/melodic/setup.bash && \
    apt-get install python-rosdistro && \
    cd ~/catkin_ws && \
    rosinstall_generator --upstream mavros | tee -a /tmp/mavros.rosinstall && \
    wstool merge -t src /tmp/mavros.rosinstall && \
    wstool update -t src && \
    apt-get install ros-melodic-geographic-msgs ros-melodic-mavlink -y && \
    chmod a+rx /opt/ros/melodic/setup.bash && \
    rosdep install --from-paths src --ignore-src -y && \
    catkin build'

RUN cd ~/catkin_ws/src && \
    git clone https://github.com/catkin/catkin_simple && \
    git clone https://github.com/ethz-asl/eigen_catkin

RUN /bin/bash -c 'cd ~/catkin_ws && \
    catkin build && \
    source ~/catkin_ws/devel/setup.bash'

RUN cd ~/catkin_ws/src && \
    git clone https://github.com/Jaeyoung-Lim/mavros_controllers.git

RUN /bin/bash -c 'source ~/catkin_ws/devel/setup.bash && \
    cd ~/catkin_ws && \
    wstool merge -t src src/mavros_controllers/dependencies.rosinstall && \
    wstool update -t src -j4 && \
    rosdep install --from-paths src --ignore-src -y --rosdistro $ROS_DISTRO && \
    catkin build && \
    source ~/catkin_ws/devel/setup.bash'

COPY mavros_offboard /root/catkin_ws/src/mavros_offboard

RUN /bin/bash -c 'cd ~/catkin_ws && \
    catkin build && \
    source ~/catkin_ws/devel/setup.bash'

# # Set the working directory
# WORKDIR /root/catkin_ws

# Install necessary packages
RUN apt-get update && apt-get install -y \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-gl \
    libqt5gui5 \
    libfuse2 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Run bash to keep the container running
CMD ["bash"]