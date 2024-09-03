FROM buildpack-deps:bookworm

ARG numpy_opts='native'
ENTRYPOINT ["dumb-init", "venv_baseline/bin/python", "-u", "quickstart.py", "--database", "/output/data.db"]
SHELL ["/usr/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    python3-dev python3-pip python3-setuptools ninja-build dumb-init sqlite3

RUN git clone --depth 1 --branch v3.12.0 https://github.com/python/cpython.git cpython-baseline
RUN git clone --depth 1 --branch v1.26.4 https://github.com/numpy/numpy.git numpy-baseline

RUN cd cpython-baseline && ./configure --prefix="$PWD"/localinstall \
  && make -j $(nproc); make install
RUN cpython-baseline/localinstall/bin/python3 -m venv venv_baseline

RUN source venv_baseline/bin/activate \
      && cd numpy-baseline \
      && git submodule update --init \
      && pip install meson-python setuptools cython \
      && pip install -r build_requirements.txt

RUN source venv_baseline/bin/activate \
  && cd numpy-baseline \
  && pip install -v . --no-build-isolation --config-settings="setup-args=-Dcpu-baseline=${numpy_opts}" --config-settings="setup-args=-Dbuildtype=release" --config-settings="build-dir=build-release"

RUN mkdir ~/.ssh
RUN ssh-keyscan github.com >> ~/.ssh/known_hosts

RUN --mount=type=ssh git clone git@github.com:fberlakovich/cmq-ae.git cpython
RUN --mount=type=ssh git clone git@github.com:fberlakovich/cmq-numpy-ae.git numpy

RUN cd cpython  \
  && ./configure --prefix="$PWD"/localinstall --with-cmlq=always \
  && make -j $(nproc); make install
RUN cpython/localinstall/bin/python3 -m venv venv_cmq

RUN source venv_cmq/bin/activate \
      && cd numpy \
      && git submodule update --init \
      && pip install attrs \
      && pip install -r build_requirements.txt

RUN source venv_cmq/bin/activate \
  && cd numpy \
  && pip install -v .  --no-build-isolation --config-settings="setup-args=-Dcpu-baseline=${numpy_opts}" --config-settings="setup-args=-Dbuildtype=release" --config-settings="build-dir=build-release"


RUN cd cpython && git worktree add ../cpython-stats HEAD
RUN cd cpython-stats && ./configure --prefix="$PWD"/localinstall --with-cmlq=always --with-instr-stats=yes \
  && make -j $(nproc); make install
RUN cpython-stats/localinstall/bin/python3 -m venv venv_cmq_stats

RUN source venv_cmq_stats/bin/activate \
      && cd numpy \
      && pip install attrs \
      && pip install -r build_requirements.txt

RUN source venv_cmq_stats/bin/activate \
  && cd numpy \
  && pip install -v . --no-build-isolation --config-settings="setup-args=-Dcpu-baseline=${numpy_opts}" --config-settings="setup-args=-Dbuildtype=release" --config-settings="setup-args=-Dcmlq-stats=true" --config-settings="build-dir=build-stats"

COPY requirements.txt ./
RUN source venv_baseline/bin/activate \
    && pip install --no-cache-dir -r requirements.txt

RUN source venv_cmq/bin/activate \
    && pip install --no-cache-dir -r requirements.txt

RUN source venv_cmq_stats/bin/activate \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir /output