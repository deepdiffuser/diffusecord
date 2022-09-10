FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get install -y libglib2.0-0 wget

ENV PATH=/root/miniconda3/bin:$PATH
RUN wget -O ~/miniconda.sh -q --show-progress --progress=bar:force https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash ~/miniconda.sh -b && rm ~/miniconda.sh

RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

RUN conda init bash

RUN pip install diffusers transformers ftfy py-cord

WORKDIR "/diffusecord"

ENTRYPOINT ["./main.py"]
