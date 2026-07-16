# Minimal GPU image for ccc-gpu, mirroring the documented conda-lock install path.
#
#   docker build -t ccc-gpu .
#   docker run --rm --gpus all ccc-gpu      # runs the install-verification one-liner
#
# The CUDA 12.5 *runtime* base provides the CUDA runtime libraries. The compiler
# toolchain required to build the extension (cuda-nvcc, cmake, gxx) is pulled in
# by the conda-lock environment, so a -runtime (not -devel) base is sufficient.
FROM nvidia/cuda:12.5.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Miniforge provides conda/mamba; conda-lock builds the reproducible environment.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bzip2 ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && conda install -n base -y conda-lock \
    && conda clean --all --yes

WORKDIR /opt/ccc-gpu

# Create the ccc-gpu environment from the committed lock file first, so this
# expensive layer is cached independently of source changes.
COPY conda-lock.yml ./
RUN conda-lock install --conda mamba --name ccc-gpu conda-lock.yml \
    && conda clean --all --yes

# Build and install the package from source (matches README "pip install .").
# Build isolation fetches scikit-build-core/cmake/ninja; nvcc + CUDA libs come
# from the conda-lock environment on PATH.
COPY . .
RUN mamba run -n ccc-gpu pip install .

# Sanity-check the install at build time via the CPU import path (a GPU is not
# available during `docker build`).
RUN mamba run -n ccc-gpu python -c "from ccc.coef.impl import ccc; print('ccc-gpu CPU import OK')"

# Default command: the documented GPU install-verification one-liner
# (requires `docker run --gpus all`).
CMD ["mamba", "run", "-n", "ccc-gpu", "python", "-c", \
     "from ccc.coef.impl_gpu import ccc as ccc_gpu; import numpy as np; print(ccc_gpu(np.random.rand(100), np.random.rand(100)))"]
