# syntax=docker/dockerfile:1.7

# ── Builder ───────────────────────────────────────────────────────────────────
# `rust:1-slim` tracks the latest stable 1.x (smaller than bookworm-full).
# The workspace's declared `rust-version = "1.80"` is a minimum floor;
# transitive deps (pem-rfc7468 1.x) require Rust edition 2024, which
# stabilized in 1.85, so the bare 1.80 toolchain fails. protobuf-src bundles
# protoc. g++/cmake/pkg-config/libssl-dev cover transitive C deps +
# reqwest's native-tls. libopenblas-dev is picked up by larql-compute on
# Linux.
FROM rust:1-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        cmake \
        build-essential \
        g++ \
        libopenblas-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

# Arch-aware CPU feature flags. The generic aarch64 / x86_64 targets don't
# enable modern SIMD extensions by default, but larql-compute uses inline
# asm (sdot on ARM, avx2/fma on x86) that requires them. Matches the
# deploy/fly/Dockerfile pattern but picks flags per TARGETARCH so the same
# image builds on both Apple Silicon podman VMs and x86 hosts.
ARG TARGETARCH
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
        amd64) export RUSTFLAGS="-C target-feature=+avx2,+fma" ;; \
        arm64) export RUSTFLAGS="-C target-feature=+dotprod" ;; \
        *)     echo "Unsupported TARGETARCH=${TARGETARCH}"; exit 1 ;; \
    esac \
    && cargo build --release -p larql-server \
    && strip target/release/larql-server \
    && cp target/release/larql-server /usr/local/bin/larql-server

# ── Runtime ───────────────────────────────────────────────────────────────────
# ubuntu:24.04 matches the glibc ABI the builder uses (slim bookworm would
# also work; ubuntu mirrors the fly deploy image for parity). libssl3 is
# required by reqwest's native-tls at runtime; ca-certificates lets
# HF/HTTPS calls verify; libopenblas0 is the runtime counterpart of
# libopenblas-dev; curl is used by the compose healthcheck.
FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        libopenblas0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/larql-server /usr/local/bin/larql-server

# Mount vindexes here. Extract jobs also land here, so the container needs
# write access to this path.
ENV LARQL_VINDEX_DIR=/vindexes
VOLUME ["/vindexes"]

# HF_TOKEN is read at runtime by the HuggingFace download / publish paths
# (see crates/larql-vindex/src/format/huggingface/publish.rs). Not baked in —
# pass via `-e HF_TOKEN=…` (docker run) or compose's `environment:` block.
# Anonymous access still works for public models when unset.

EXPOSE 8080

# --cors is required for the UI (any origin) to hit this server. Host binds to
# 0.0.0.0 by default; no extra flag needed.
ENTRYPOINT ["larql-server"]
CMD ["--cors"]
