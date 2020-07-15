ARG CUDA="10.2"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04
# FROM pytorch/pytorch:latest

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
    && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
    && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev wget 

# Install Miniconda
RUN cd / && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x /Miniconda3-latest-Linux-x86_64.sh \
    && /Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda \
    && rm /Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install -y conda-build \
    && /miniconda/bin/conda create -y --name py37 python=3.7 \
    && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install --default-timeout=1000 numpy ninja yacs cython  matplotlib opencv-python tqdm pyyaml tensorboardX hiddenlayer -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

# install pytorch
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
# RUN conda config --set show_channel_urls yes && conda install pytorch torchvision cudatoolkit=${CUDA} -c pytorch \
#     && conda clean -ya
# RUN conda install pytorch torchvision cudatoolkit=${CUDA} -c pytorch \
#     && conda clean -ya

# set cuda path 
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CPATH=/usr/local/cuda/include:$CPATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

RUN pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install torch-geometric -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install torchvision==0.6.0 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

WORKDIR /meshgraph
