FROM mxnet/python:1.3.0_gpu_cu90_py3

RUN pip install gluoncv --pre --upgrade
RUN pip install -U jupyter && pip install jupyterlab

# 安装opencv所需要的一些依赖
RUN apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext-dev 
RUN pip install opencv-python pandas

# separate the base layer here for speeding up the docker build as modify source code
CMD [ "bash" ]
