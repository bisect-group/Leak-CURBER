# hash:sha256:6b51d31ef1e4ba5d291c0783e8946ba73f56420aba11f3bcfcb9bcf3e23e60f1
FROM registry.codeocean.com/codeocean/ubuntu:20.04-cuda11.7.0-cudnn8

ARG DEBIAN_FRONTEND=noninteractive

ENV PATH=/root/miniforge3/bin:/usr/local/cuda-11.3/bin:$PATH
ENV CPATH=/usr/local/cuda-11.3/include:$CPATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-libraries-11-3=11.3.1-1 \
        cuda-libraries-dev-11-3=11.3.1-1 \
        cuda-toolkit-11-3=11.3.1-1 \
        wget=1.20.3-1ubuntu2.1 \
    && rm -rf /var/lib/apt/lists/*

COPY postInstall /
RUN /postInstall
