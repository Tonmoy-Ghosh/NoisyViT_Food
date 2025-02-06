# Getting base image
#FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3
FROM nvcr.io/nvidia/tensorflow:22.05-tf2-py3

### Working directory, tmp is the folder inside docker image, all files copy there from local drive to tmp
WORKDIR /workercode
COPY . .


###### Python 3.8.12 Install from makefile ################
RUN apt-get update -y

RUN echo y |apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev liblzma-dev

RUN wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz

RUN tar -xf Python-3.8.12.tgz

WORKDIR "Python-3.8.12"

RUN ./configure --enable-optimizations 

RUN make install

WORKDIR /workercode

RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.8 1
RUN rm -rf "Python-3.8.12"
RUN rm -f "Python-3.8.12.tgz"
###################### finish python 3.8.12 install ######

RUN echo y | apt-get install python-tk

RUN apt-get update -y && apt-get install -y libmariadb-dev libmariadb3 gcc

#RUN echo y | apt-get install python3-pip

RUN python -m pip install --upgrade pip

#####RUN apt install libmariadb3 libmariadb-dev

RUN apt-get update -y && apt-get install ffmpeg libsm6 libxext6  -y bash coreutils grep sed  build-essential bash coreutils grep sed python-all-dev libexiv2-dev libopencv-dev python3-opencv


RUN apt-get install bash coreutils grep sed

### This will need  in future for YOLO
#RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html 
#RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html 
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install -r requirements.txt

# ###this need to keep separate###

RUN pip3 install mysql-connector-python-rf==2.2.2


####environment variable to indicate running in Docker
ENV AM_I_IN_A_DOCKER_CONTAINER Yes

#WORKDIR /workercode/Grounded-SAM
#RUN python -m pip install --no-cache-dir -e segment_anything
#RUN python -m pip install --no-cache-dir wheel
#RUN python -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO
#WORKDIR /workercode
#CMD [ "python","-u","worker.py"]