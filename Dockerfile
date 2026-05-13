# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      clang \
      cmake \
      ninja-build \
      pkg-config \
      git \
      ca-certificates \
      libwebp-dev \
      libeigen3-dev \
      nlohmann-json3-dev \
      libabsl-dev \
      zlib1g-dev

WORKDIR /workspace
