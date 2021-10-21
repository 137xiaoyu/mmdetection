ARG PYTORCH="1.7.0"
ARG CUDA="11.0"
ARG CUDNN="8"
FROM image.rsaicp.com/base/cuda:11.0-cudnn8-runtime-ubuntu18.04
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"


RUN apt-get clean
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 swig --fix-missing \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# install mmcv-full
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

# install shapely
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple shapely

COPY . /work

# build DOTA_devkit
RUN cd /work/DOTA_devkit && swig -c++ -python polyiou.i && python setup.py build_ext --inplace

# build mmdet
RUN cd /work && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements/build.txt && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .

# build apex
# RUN cd /work/apex && pip install --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# 容器启动命令
CMD ["python3", "-u", "/work/main.py"]