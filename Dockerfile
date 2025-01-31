FROM nvidia/cuda:11.0-devel-ubuntu18.04 AS cuda-builder

ENV APP_NAME="lungs"
ENV USER=root
ENV HOME=/root

USER root

ENV PYTHONPATH=/home/${USERNAME}/$APP_NAME:/home/${USERNAME}

RUN apt-get -y update
RUN apt-get  install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
	build-essential \
        curl \
        gcc \
        g++ \
        python3.8-dev \
        python3-pip \
        python3.8 \
        libopenmpi-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglib2.0-0

RUN apt-get install -y ffmpeg
RUN apt-get install -y zip

RUN python3.8 -m pip install pip

RUN python3.8 -m pip install --upgrade pip

COPY  ./torch-1.10.0+cu111-cp38-cp38-linux_x86_64.whl ./torch-1.10.0+cu111-cp38-cp38-linux_x86_64.whl
RUN python3.8 -m pip install torch-1.10.0+cu111-cp38-cp38-linux_x86_64.whl

RUN python3.8 -m pip install flask
RUN python3.8 -m pip install werkzeug

RUN python3.8 -m pip install pyyaml
RUN python3.8 -m pip install redis
RUN python3.8 -m pip install schedule
RUN python3.8 -m pip install pydicom
RUN python3.8 -m pip install pynetdicom==1.5.7

RUN python3.8 -m pip install setuptools==45.2.0
RUN python3.8 -m pip install uwsgi

RUN python3.8 -m pip install requests

RUN python3.8 -m pip install cython

RUN python3.8 -m pip install numpy
RUN python3.8 -m pip install opencv-python==4.1.2.30

RUN python3.8 -m pip install matplotlib
RUN python3.8 -m pip install mysql-connector-python

RUN python3.8 -m pip install rq

RUN python3.8 -m pip install Flask-Session

RUN python3.8 -m pip install Flask-Cors

RUN python3.8 -m pip install SimpleITK==2.0.2

RUN python3.8 -m pip install pysftp
RUN python3.8 -m pip install dicom2nifti

RUN python3.8 -m pip install opencv-python
RUN python3.8 -m pip install pyrebase4
RUN python3.8 -m pip install firebase_admin

RUN python3.8 -m pip install awscli

RUN python3.8 -m pip install pandas

RUN python3.8 -m pip install scikit-image
RUN python3.8 -m pip install werkzeug==2.0.0

COPY ./k3d-2.12.0-py2.py3-none-any.whl ./k3d-2.12.0-py2.py3-none-any.whl
RUN python3.8 -m pip install k3d-2.12.0-py2.py3-none-any.whl

RUN python3.8 -m pip install webpage2html lxml mako pyacvd glob2 termcolor

RUN python3.8 -m pip install python-gdcm

RUN python3.8 -m pip install docxtpl

RUN python3.8 -m pip install docx2pdf



# libreoffice

RUN apt-get -y update || true
RUN apt-get install -y wget

RUN apt-key del 7fa2af80
RUN rm /etc/apt/sources.list.d/cuda.list
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb 
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && \
	apt-get -y -q install libreoffice libreoffice-writer ure libreoffice-java-common libreoffice-core libreoffice-common openjdk-8-jre \
		fonts-opensymbol hyphen-fr 	hyphen-de hyphen-en-us hyphen-it hyphen-ru 	fonts-dejavu fonts-dejavu-core \
		fonts-dejavu-extra 	fonts-droid-fallback fonts-dustin fonts-f500 fonts-fanwood 	fonts-freefont-ttf 	fonts-liberation fonts-lmodern \
		fonts-lyx fonts-sil-gentium fonts-texgyre fonts-tlwg-purisa && \
	apt-get -y -q remove libreoffice-gnome && \
	apt -y autoremove && \
	rm -rf /var/lib/apt/lists/*

RUN adduser --home=/opt/libreoffice --disabled-password --gecos "" --shell=/bin/bash libreoffice

# vim
RUN apt-get -y update || true
RUN apt-get install -y vim

# chrome
RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt-get install -y ./google-chrome-stable_current_amd64.deb

COPY ./chromedriver /usr/bin/chromedriver

RUN python3.8 -m pip install selenium webdriver_manager


COPY ./${APP_NAME}/ ${APP_NAME}/

RUN ln -s /usr/bin/python3.8 /usr/bin/python

WORKDIR /${APP_NAME}/

