FROM hardikparwana/cuda116desktop:ipopt
RUN apt-get update
RUN apt install -y python3-pip
RUN apt install -y vim tmux
#RUN sudo apt-get install gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev

RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository universe
RUN apt update && apt install -y curl
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update
RUN apt install -y ros-galactic-desktop
RUN apt install -y ros-dev-tools
RUN echo 'source /opt/ros/galactic/setup.bash' >> ~/.bashrc

