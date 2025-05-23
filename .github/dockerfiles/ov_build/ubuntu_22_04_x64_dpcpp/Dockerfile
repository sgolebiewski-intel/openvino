ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:22.04

USER root

# APT configuration
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::Fix-Broken "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::no-install-recommends "true";' >> /etc/apt/apt.conf

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

RUN apt-get update && \
    apt-get install software-properties-common && \
    add-apt-repository --yes --no-update ppa:git-core/ppa && \
    add-apt-repository --yes --no-update ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install \
        wget \
        curl \
        git \
        ca-certificates \
        gpg-agent \
        tzdata \
        libtbb2 \
        # parallel gzip
        pigz \
        # Pythons \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        default-jdk \
        && \
    rm -rf /var/lib/apt/lists/*


# Install OneAPI Toolkit
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update && \
    apt-get install \
        intel-oneapi-compiler-dpcpp-cpp=2024.2.1-1079 \
        && \
    rm -rf /var/lib/apt/lists/*

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    bash -e /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    python3.11 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Use Python 3.11 as default
# Using venv here 'cause other methods to switch the default Python on Ubuntu break both system and wheels build
RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$SCCACHE_HOME:$PATH"

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}

# OneAPI env
ENV ONEAPI_ROOT=/opt/intel/oneapi
ENV PKG_CONFIG_PATH=/opt/intel/oneapi/compiler/2024.2/lib/pkgconfig
ENV DIAGUTIL_PATH=/opt/intel/oneapi/dpcpp-ct/2024.2/etc/dpct/sys_check/sys_check.sh:/opt/intel/oneapi/debugger/2024.2/etc/debugger/sys_check/sys_check.py:/opt/intel/oneapi/compiler/2024.2/etc/compiler/sys_check/sys_check.sh
ENV MANPATH=/opt/intel/oneapi/debugger/2024.2/share/man:/opt/intel/oneapi/compiler/2024.2/share/man:
ENV GDB_INFO=/opt/intel/oneapi/debugger/2024.2/share/info/
ENV CMAKE_PREFIX_PATH=/opt/intel/oneapi/compiler/2024.2
ENV CMPLR_ROOT=/opt/intel/oneapi/compiler/2024.2
ENV INFOPATH=/opt/intel/oneapi/debugger/2024.2/share/info
ENV LIBRARY_PATH=/opt/intel/oneapi/compiler/2024.2/lib
ENV OCL_ICD_FILENAMES=/opt/intel/oneapi/compiler/2024.2/lib/libintelocl.so
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/debugger/2024.2/opt/debugger/lib:/opt/intel/oneapi/compiler/2024.2/opt/compiler/lib:/opt/intel/oneapi/compiler/2024.2/lib
ENV NLSPATH=/opt/intel/oneapi/mkl/2024.2/share/locale/%l_%t/%N:/opt/intel/oneapi/compiler/2024.2/lib/compiler/locale/%l_%t/%N
ENV PATH=$PATH:/opt/intel/oneapi/dev-utilities/2024.2/bin:/opt/intel/oneapi/debugger/2024.2/opt/debugger/bin:/opt/intel/oneapi/compiler/2024.2/bin
ENV INTEL_PYTHONHOME=/opt/intel/oneapi/debugger/2024.2/opt/debugger
ENV CPATH=/opt/intel/oneapi/dpl/2022.6/include:/opt/intel/oneapi/dev-utilities/2024.2/include:

# Set Intel DPC++ as a default compiler
ENV CC=icx
ENV CXX=icpx
