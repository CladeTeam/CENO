ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.15.1
FROM ${VLLM_BASE_IMAGE}
WORKDIR /workspace/CENO
COPY . /workspace/CENO
RUN python3 -c "from importlib.metadata import version; assert version('vllm') == '0.15.1'"
