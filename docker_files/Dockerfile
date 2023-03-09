# Need this cuda image
FROM hardikparwana/cuda116desktop:ipopt
RUN apt-get update
RUN apt install -y python3-pip
RUN pip3 install numpy==1.22.3 matplotlib sympy argparse scipy==1.8.0
RUN pip3 install torch==1.12.1 torchvision==0.13.1
RUN apt-get -y install vim
RUN pip3 install cvxpy==1.2.0 cvxpylayers==0.1.5 gpytorch
RUN apt -y install ffmpeg
RUN pip3 install gym==0.26.0 gym-notices==0.0.8 gym-recording==0.0.1 moviepy==1.0.3 pygame==2.1.2
RUN apt-get install pkg-config
RUN pip3 install cyipopt==1.2.0


