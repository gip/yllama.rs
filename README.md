# yllama.rs

The idea was to work on a non-trivial implementation to learn a bit of Rust and get back into coding after years of engineering management. Project was timeboxed to a few days. Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp), I set the goal to deliver a Llama 3 8b inference implementation that could run on a modern laptop and could also be deployed to the Internet Computer (ICP). Was fun.

## Goals
Functional goals
* Llama 3 8b inference on laptop and ICP with maximum code reuse between the two targets - that also means that the code needed to be modular to be able to be deployed on ICP canisters
* Learn transformers internals
* Support [GGUF files](https://huggingface.co/docs/hub/en/gguf)
* Support some form of model quantization
Non-functional goals
* Pure Rust as it is well supported to build on ICP
* Built from scratch to maximize learning, so I didn't use any of [Candle](https://github.com/huggingface/candle)
* No dynamic dispatch or checks during model execution - model statically built including for value initialization (I regretted that choice!)
* Naive implementation, leaving optimization as a later act

## Usage
* F32 and F16-quantized tensors are supported. A [GGUF file](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-fp32.gguf) can be downloaded from Hugging Face.
* Hugging Face [`tokenizers`](https://docs.rs/tokenizers/0.19.1/tokenizers/) is currently used but will be replaced by a custom implementation. For now a tokenized file needs to be provided. For instance [this file](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer.json) for LLama 3. 

To start:
> cargo run --release -- -f ../Meta-Llama-3-8B-Instruct/ggml-model-f32.gguf -t ../llama-3-tokenizer/tokenizer.json -p "What is the capital of France?"

* Generation speed is around 1 token / second depending on memory
* For the deployment on ICP, please refer to this [repo](https://github.com/gip/yllama.oc)

## Learnings
* Rust is a pretty neat language with great library and superior tooling and I _felt_ productive quickly (which doesn't mean I was)
* The beginner channel of the Discord Rust community was an amazing resoource
* Typing in Rust is more limited and verbose than Haskell and that slowed my down considerably at some point. A lot of typing decisions I took were probably wrong (the [llama.rs](https://github.com/gip/yllama.rs/blob/main/yllama/src/llama.rs) is an eyesore!)
* The inner matmul loops in both aarch64 and wasm are relatively well optimized out of the box in release mode (no SIMD though) - Rust optimized seems adequate
* Gpt/Claude were not really able to help much

## Reference
* [Meta Llama 3 8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model
* [llama.cpp](https://github.com/ggerganov/llama.cpp) and [llama3.c](https://github.com/jameswdelancey/llama3.c)

