FROM ubuntu:16.04

LABEL authors="Romain Michelucci (r.michelucci@campus.unimib.it)" 
LABEL Description="Dockerised Simulation of Urban MObility(SUMO) and experiments"

ENV SUMO_VERSION 1.8.0
ENV SUMO_HOME /opt/sumo
ENV SUMO_USER romain

# Install system dependencies.
RUN apt-get update && apt-get -qq install \
    wget \
    g++ \
    make \
    libxerces-c-dev \
    libfox-1.6-0 libfox-1.6-dev \
    python2.7

# Download and extract source code
RUN wget http://downloads.sourceforge.net/project/sumo/sumo/version%20$SUMO_VERSION/sumo-src-$SUMO_VERSION.tar.gz
RUN tar xzf sumo-src-$SUMO_VERSION.tar.gz && \
    mv sumo-$SUMO_VERSION $SUMO_HOME && \
    rm sumo-src-$SUMO_VERSION.tar.gz

# Configure and build from source.
RUN cd $SUMO_HOME && ./configure && make install

RUN adduser $SUMO_USER --disabled-password
# CMD sumo-gui


## INSTALLATIONS
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip python3-dev wget \
    bzip2 libopenblas-dev pbzip2 libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# INSTALLATION OF CONDA
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# Install git
RUN apt-get update \        
     apt-get install -y git
     
      
RUN mkdir /home/test \      
           cd /home/test \        
           git clone https://github.com/GameDisplayer/Deep-QLearning-Multi-Agent-Perpsective-for-Traffic-Signal-Control.git

#Copy env
RUN cp Deep-QLearning-Multi-Agent-Perpsective-for-Traffic-Signal-Control/2TLCS/environment.yml .

ADD environment.yml /tmp/environment.yml

RUN apt-get update

RUN conda env create -f /tmp/environment.yml # Pull the environment name out of the environment.yml

RUN ln -sf /bin/bash /bin/sh #make sure to use bash instead of zsh

RUN apt-get update && \
      apt-get -y install sudo 

#Set working directory
WORKDIR /home/test/Deep-QLearning-Multi-Agent-Perpsective-for-Traffic-Signal-Control/2TLCS

RUN conda init && \
    source /home/test/.bashrc && \
    source activate tf_gpu && \
    echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > /home/test/.bashrc

ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

CMD /bin/bash