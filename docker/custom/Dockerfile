# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.06-py3

# Uncomment this and below to create a different conda environment
# Miniconda env Installation in a different environment
# FROM continuumio/miniconda3 as miniconda
# COPY environment.yaml .
# RUN conda env create -f environment.yaml

# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "DeepLearningExamples", "/bin/bash", "-c"]

# RUN echo "Test conda environment installation"
# RUN python -c "import tensorboardX"

FROM nvcr.io/nvidia/tritonserver:20.06-v1-py3-clientsdk as trt
FROM ${FROM_IMAGE_NAME}
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract neovim zsh

ENV BERT_PREP_WORKING_DIR /workspace/bert/data

WORKDIR /workspace
# Copy the perf_client over
COPY --from=trt /workspace/install/ /workspace/install/
ENV LD_LIBRARY_PATH /workspace/install/lib:${LD_LIBRARY_PATH}

# Install trt python api
RUN apt-get install libb64-0d
RUN pip install /workspace/install/python/tensorrtserver*.whl

WORKDIR /workspace/bert
COPY . .
RUN pip install datasets
RUN pip install --upgrade --no-cache-dir pip \
	&& pip install --no-cache-dir -e .

RUN apt-get install -y iputils-ping

# Manually install conda packages
RUN conda install -c conda-forge click=7.1.1 \
	fzf \
	jupyter_console=6.2.0 \
	pudb=2019.2 \
	tensorboardX=2.1 \
	tensorflow=1.14.0

# Currently downloading cpu version of mxnet because cuda version 11 not supported by mxnet for gpu
RUN pip install \
    loguru==0.4.1 \
    click-config-file==0.5.0 \
    mxnet==1.7.0.post1 \
    gluonnlp==0.8.3 \
    wandb==0.10.7

# Install Oh-my-zsh
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/deluan/zsh-in-docker/master/zsh-in-docker.sh)" -- \
    -t agnoster \
		-p fzf \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
		-p https://github.com/zsh-users/zsh-history-substring-search \
		-p https://github.com/zsh-users/zsh-syntax-highlighting
RUN echo "alias vim=nvim" >> $HOME/.zshrc
RUN echo "export FZF_BASE=\"$HOME/.fzf/shell\"" >> $HOME/.zprofile

RUN echo "exec zsh" >> $HOME/.zprofile

# Uncomment this and above if you want to activate another environment upon start up
# ENV PATH /opt/conda/bin/:$PATH
# RUN echo "source activate DeepLearningExamples" >> $HOME/.zshrc

RUN echo "Checking installation"
RUN python -c "import apex; import transformers"
RUN echo "Successfully installed!"

ARG wandb_api_key
RUN wandb login ${wandb_api_key}

ENTRYPOINT ["/usr/bin/zsh", "--login"]

