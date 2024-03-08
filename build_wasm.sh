#!/bin/bash

RUSTFLAGS=--cfg=web_sys_unstable_apis cargo build --release --target wasm32-unknown-unknown --lib
wasm-bindgen --out-dir target/generated --web target/wasm32-unknown-unknown/release/physarum.wasm