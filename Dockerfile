FROM vllm/vllm-openai:latest

# Python tooling + small runtime deps

# Install build dependencies using the base image sources.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    pkg-config \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade \
    pip \
    setuptools \
    wheel \
    packaging \
    ninja

RUN python3 -m pip install --ignore-installed --no-cache-dir \
    sentencepiece==0.1.99 \
    flask-restful

RUN cd /tmp && \
    git clone https://github.com/Dao-AILab/causal-conv1d.git && \
    cd causal-conv1d && \
    git checkout v1.2.2.post1 && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE python3 -m pip install --no-build-isolation --no-cache-dir . && \
    cd / && rm -rf /tmp/causal-conv1d

RUN cd /tmp && \
    git clone https://github.com/state-spaces/mamba.git && \
    cd mamba && \
    git checkout v2.0.3 && \
    MAMBA_FORCE_BUILD=TRUE python3 -m pip install --no-build-isolation --no-cache-dir . && \
    cd / && rm -rf /tmp/mamba

RUN python3 -c "import causal_conv1d, mamba_ssm; print('ok')"
