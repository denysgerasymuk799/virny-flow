FROM ubuntu:20.04

WORKDIR /auto-sklearn

# Install linux packages
RUN apt-get update

# Set the locale
# workaround for https://github.com/automl/auto-sklearn/issues/867
RUN apt-get -y install locales
RUN touch /usr/share/locale/locale.alias
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Set environment variables to use only one core
RUN export OPENBLAS_NUM_THREADS=1
RUN export MKL_NUM_THREADS=1
RUN export BLAS_NUM_THREADS=1
RUN export OMP_NUM_THREADS=1

# Install build requirements
RUN apt install -y python3-dev python3-pip
RUN pip3 install --upgrade setuptools

# Avoid tzdata interactive prompt
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    tzdata \
    software-properties-common \
    git \
    build-essential \
    swig && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Add deadsnakes PPA and install Python 3.9 + pip
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils

# Set python3 and pip3 to point to python3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Copy the checkout autosklearn version for installation
ADD . /auto-sklearn/

# Upgrade pip then install dependencies
RUN pip3 install --upgrade pip

# Install
RUN pip3 install python-dotenv~=1.0.1

RUN pip3 install pymongo~=4.8.0

RUN pip3 install xgboost lightgbm aif360[Reductions] tqdm folktables munch PyYAML colorama

RUN pip3 install -r requirements.txt
