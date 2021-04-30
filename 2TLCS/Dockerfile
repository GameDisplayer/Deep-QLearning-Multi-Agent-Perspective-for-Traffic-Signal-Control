FROM ubuntu:20.04

USER root

LABEL Description="Dockerised Simulation of Urban MObility(SUMO)"

ENV SUMO_VERSION 1.8.0
ENV SUMO_HOME /opt/sumo
ENV SUMO_USER romain

# Install system dependencies.
RUN apt-get update && \
    apt-get install --no-install-recommends -y python3.8 python3-pip python3.8-dev
RUN apt-get install -y software-properties-common
RUN apt-get install -y cmake g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig sudo wget

#RUN python3.8 --version

# Download and extract source code
RUN wget http://downloads.sourceforge.net/project/sumo/sumo/version%20$SUMO_VERSION/sumo-src-$SUMO_VERSION.tar.gz
RUN tar xzf sumo-src-$SUMO_VERSION.tar.gz && \
    mv sumo-$SUMO_VERSION $SUMO_HOME && \
    rm sumo-src-$SUMO_VERSION.tar.gz

# download the tar.gz from the official site
#ADD sumo-src-$SUMO_VERSION.tar.gz .

#RUN mv sumo-$SUMO_VERSION $SUMO_HOME 

RUN cd $SUMO_HOME && export SUMO_HOME="$PWD" && mkdir -p ./build/cmake-build && cd build/cmake-build && cmake ../.. && make -j$(nproc)
ENV PATH=$PATH:$SUMO_HOME/bin 

#Requirements
WORKDIR /experiment
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install traci

CMD /bin/bash
