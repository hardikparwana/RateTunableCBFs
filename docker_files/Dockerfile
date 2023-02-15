# Need this cuda image
FROM hardikparwana/cuda116desktop:v1
RUN apt-get update
RUN apt install -y python3-pip
RUN apt install -y vim tmux

################################ IPOPT Installation ################################################

# Dependency 1
RUN apt-get -y install gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev

# Dependency 2
WORKDIR /home
RUN mkdir softwares 
WORKDIR /home/softwares
RUN git clone https://github.com/coin-or-tools/ThirdParty-ASL.git
WORKDIR /home/softwares/ThirdParty-ASL
RUN ./get.ASL
RUN ./configure
RUN make
RUN make install

# Dependency 3
RUN apt-get update & apt-get install liblapack-dev

# Dependency 4
WORKDIR /home/softwares
RUN git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git
WORKDIR /home/softwares/ThirdParty-Mumps
RUN ./get.Mumps
RUN ./configure
RUN make
RUN make install

# Dependency 5
WORKDIR /home/softwares
RUN git clone https://github.com/coin-or-tools/ThirdParty-HSL.git
WORKDIR /home/softwares/ThirdParty-HSL
RUN wget https://www.dropbox.com/s/6vgc0ndjtuio05a/coinhsl.zip?dl=0
RUN mv coinhsl.zip* coinhsl.zip
RUN unzip coinhsl.zip
RUN rm coinhsl.zip
RUN ./configure
RUN make
RUN make install

# Now finally Ipopt
WORKDIR /home/softwares
RUN git clone https://github.com/coin-or/Ipopt.git
WORKDIR /home/softwares/Ipopt
RUN mkdir build
WORKDIR /home/softwares/Ipopt/build
RUN ../configure
RUN make
#RUN make test
RUN make install


WORkDIR /home


############################################ ROS Installation ############################################

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

