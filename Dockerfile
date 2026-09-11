FROM ubuntu:24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_REF=v0.4.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggml-org/llama.cpp.git

WORKDIR /opt/llama.cpp
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --config Release --target llama-server -j"$(nproc)"

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY scripts/serve.sh /app/scripts/serve.sh

RUN chmod +x /app/scripts/serve.sh

ENV LLAMA_BIN=/usr/local/bin/llama-server \
    LLAMA_MODEL=/models/tiny-llama-f16.gguf \
    LLAMA_HOST=0.0.0.0 \
    LLAMA_PORT=8080 \
    LLAMA_CONTEXT_SIZE=1024

EXPOSE 8080
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 CMD ["/bin/sh", "-c", "curl --fail --silent http://127.0.0.1:${LLAMA_PORT}/v1/models >/dev/null"]

ENTRYPOINT ["/app/scripts/serve.sh"]
