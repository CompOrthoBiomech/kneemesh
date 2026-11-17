# The kneemesh GUI, kneemesh-gui, served on port 8080, with the console scripts alongside.
#
#   docker build -t kneemesh .
#   docker run --rm -p 8080:8080 -v "$PWD:/data" kneemesh
#
# then open http://localhost:8080. /data is the working directory, where the GUI writes its outputs.
# The build stage installs the locked runtime dependencies into /opt/kneemesh and prunes them with
# docker/prune.py, which keeps only the shared objects a traced run of the pipelines and GUI loads.

ARG PYTHON_VERSION=3.13

# Shared libraries the wheels need. gmsh links OpenMP, GLU, GL, and X client libraries, though it never opens
# a display here; VTK renders offscreen through EGL on Mesa's software rasterizer (llvmpipe).
ARG RUNTIME_PACKAGES="libgomp1 libglu1-mesa libgl1 libegl1 libegl-mesa0 libgl1-mesa-dri libxrender1 libxcursor1 libxft2 libxinerama1"

FROM python:${PYTHON_VERSION}-slim AS build
ARG RUNTIME_PACKAGES
RUN apt-get update \
    && apt-get install -y --no-install-recommends ${RUNTIME_PACKAGES} binutils \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/kneemesh \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0
# VTK requires matplotlib only for its mathtext rendering, which kneemesh never draws; leaving it and
# its dependencies out saves about 70 MB.
ARG SYNC="uv sync --locked --no-dev --extra gui --no-editable \
    --no-install-package matplotlib --no-install-package contourpy --no-install-package cycler \
    --no-install-package fonttools --no-install-package kiwisolver --no-install-package pillow \
    --no-install-package pyparsing --no-install-package python-dateutil"

WORKDIR /src
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv ${SYNC} --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv ${SYNC}

COPY docker ./docker
ENV VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow
RUN /opt/kneemesh/bin/python docker/prune.py

FROM python:${PYTHON_VERSION}-slim
ARG RUNTIME_PACKAGES
RUN apt-get update \
    && apt-get install -y --no-install-recommends ${RUNTIME_PACKAGES} \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 kneemesh \
    && mkdir /data && chown kneemesh /data

COPY --from=build /opt/kneemesh /opt/kneemesh
ENV PATH=/opt/kneemesh/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow

# Check the pruned environment still runs the pipelines and the GUI, then drop the caches and
# preferences Mesa and gmsh's FLTK wrote.
RUN --mount=type=bind,source=docker/trace.py,target=/tmp/trace.py \
    HOME=/tmp/home python /tmp/trace.py > /dev/null \
    && rm -rf /tmp/home /etc/fltk

USER kneemesh
WORKDIR /data
EXPOSE 8080
CMD ["kneemesh-gui", "--server", "--host", "0.0.0.0", "--port", "8080"]
