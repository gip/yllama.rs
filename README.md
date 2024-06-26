# yllama.rs

Welcome to Y Llama!

## Y Llama is a weekend project with the following aims
* Learn transformer internals
* Learn Rust as this is the author's first project in this language
* Pure Rust
* No dynamic dispatch or checks during model execution - correct model statically built
* Support Llama3
* Support GGUF file
* Support some form of quantization
* Naive implementation, leaving optimization as a later act

## Usage
* As of know, only f32 tensors are supported with quantitized version coming soon (dequantization code is untested) and a [GGUF file](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-fp32.gguf) is required.
* Hugging Face [`tokenizers`](https://docs.rs/tokenizers/0.19.1/tokenizers/) is currently used but will be replaced by a custom implementation. For now a tokenized file needs to be provided. For instance [this file](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer.json) for LLama 3. 

To start:
> cargo run --release -- -f ../Meta-Llama-3-8B-Instruct/ggml-model-f32.gguf -t ../llama-3-tokenizer/tokenizer.json -p "What is the capital of France?"

