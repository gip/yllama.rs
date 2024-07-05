pub mod gpt;
pub mod llama;
pub use gpt::Gpt;
pub mod llm;

use anyhow::anyhow;
use clap::Parser;
use half::f16;
use num_traits::float::Float;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::str;

use llama::{Llama, LlamaParams};
use llm::{Instantiable, LLM};
use yloader::*;
use yloader::{load_build, load_fast, ModelFile};
use ymath::tensor::*;

pub struct VIRTUALMEM; // Basically MacOS, Linux, Windows

impl<'a, T: Float, const D0: usize, const D1: usize> Instantiable<VIRTUALMEM, (usize, String)>
    for Tensor<'a, true, T, M<D0, D1>, VecStore<T>>
{
    fn instantiate(_: (usize, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        Ok(TensorMut::new_matrix())
    }
}

impl<'a, T: Float, const D0: usize> Instantiable<VIRTUALMEM, (usize, String)>
    for Tensor<'a, true, T, V<D0>, VecStore<T>>
{
    fn instantiate(_: (usize, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        Ok(TensorMut::new_vector())
    }
}

impl<'a, const D0: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, String)>
    for Tensor<'a, false, f32, V<D0>, VecStore<f32>>
{
    fn instantiate((model, name): (&'a ModelFile, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize, const D1: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, String)>
    for Tensor<'a, false, f32, M<D0, D1>, VecStore<f32>>
{
    fn instantiate((model, name): (&'a ModelFile, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, usize, String)>
    for Tensor<'a, false, f32, V<D0>, VecStore<f32>>
{
    fn instantiate((model, i, name): (&'a ModelFile, usize, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let name = name.replace("{}", &i.to_string());
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize, const D1: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, usize, String)>
    for Tensor<'a, false, f32, M<D0, D1>, VecStore<f32>>
{
    fn instantiate((model, i, name): (&'a ModelFile, usize, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let name = name.replace("{}", &i.to_string());
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize, const D1: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, usize, String)>
    for Tensor<'a, false, f32, M<D0, D1>, VecStore<f16>>
{
    fn instantiate((model, i, name): (&'a ModelFile, usize, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let name = name.replace("{}", &i.to_string());
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize, const D1: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, String)>
    for Tensor<'a, false, f32, M<D0, D1>, VecStore<f16>>
{
    fn instantiate((model, name): (&'a ModelFile, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize, const D1: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, usize, String)>
    for Tensor<'a, false, f32, M<D0, D1>, MmapStore<f32, f32>>
{
    fn instantiate((model, i, name): (&'a ModelFile, usize, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let name = name.replace("{}", &i.to_string());
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize, const D1: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, String)>
    for Tensor<'a, false, f32, M<D0, D1>, MmapStore<f32, f32>>
{
    fn instantiate((model, name): (&'a ModelFile, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, String)>
    for Tensor<'a, false, f32, V<D0>, MmapStore<f32, f32>>
{
    fn instantiate((model, name): (&'a ModelFile, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, usize, String)>
    for Tensor<'a, false, f32, V<D0>, MmapStore<f32, f32>>
{
    fn instantiate((model, i, name): (&'a ModelFile, usize, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let name = name.replace("{}", &i.to_string());
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl Instantiable<VIRTUALMEM, &ModelFile> for LlamaParams<f32> {
    fn instantiate(model: &ModelFile) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let header = &model.header;
        let embedding_length = header_find_usize(header, "llama.embedding_length")?;
        let attention_head_count_kv = header_find_usize(header, "llama.attention.head_count_kv")?;
        let attention_head_count = header_find_usize(header, "llama.attention.head_count")?;
        let params: LlamaParams<f32> = LlamaParams {
            block_count: header_find_usize(header, "llama.block_count")?,
            _context_length: header_find_usize(header, "llama.context_length")?,
            embedding_length,
            feed_forward_length: header_find_usize(header, "llama.feed_forward_length")?,
            attention_head_count,
            attention_head_count_kv,
            attention_layer_norm_rms_epsilon: header_find_f32(
                header,
                "llama.attention.layer_norm_rms_epsilon",
            )?,
            rope_freq_base: header_find_f32(header, "llama.rope.freq_base")?,
            _rope_dimension_count: header_find_usize(header, "llama.rope.dimension_count")?,
            vocab_size: header_find_usize(header, "llama.vocab_size")?,
            _max_seq_len: header_find_usize(header, "llama.context_length")?,
            _attention_kv_length: embedding_length * attention_head_count_kv / attention_head_count,
        };
        Ok(params)
    }
}

fn process(
    path: &str,
    tokenizer_path: &str,
    prompt: &str,
    _clone: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (arch, name, gguf) = load_fast(path)?;

    println!("Architecture == {}", arch);
    println!("Name == '{}'", name);

    match arch.as_str() {
        "llama" => {
            let model = load_build(path, gguf)?;
            type A = MmapStore<f32, f32>;
            //type B = MmapStore<f32, f16>;
            type C = VecStore<f32>;
            type D = VecStore<f16>;
            let typ = llama::llama_find_type(&model)?;
            const EMBED: usize = 4096;
            const VOCAB: usize = 128256;
            const FF: usize = 14336;
            const KV: usize = 1024;
            const CONTEXT: usize = 2048;
            match typ {
                "F16" => {
                    type LlamaType<'a> = Llama<
                        'a,
                        VIRTUALMEM,
                        ModelFile,
                        f32,
                        D,
                        D,
                        C,
                        D,
                        D,
                        D,
                        C,
                        D,
                        D,
                        C,
                        D,
                        D,
                        EMBED,
                        VOCAB,
                        FF,
                        KV,
                        CONTEXT,
                    >;
                    let mut runnable: LlamaType = Llama::instantiate((&model, tokenizer_path))?;
                    unsafe { runnable.run(prompt) }
                }
                "F32" => {
                    type LlamaType<'a> = Llama<
                        'a,
                        VIRTUALMEM,
                        ModelFile,
                        f32,
                        A,
                        A,
                        A,
                        A,
                        A,
                        A,
                        A,
                        A,
                        A,
                        A,
                        A,
                        A,
                        EMBED,
                        VOCAB,
                        FF,
                        KV,
                        CONTEXT,
                    >;
                    let mut runnable: LlamaType =
                        Instantiable::instantiate((&model, tokenizer_path))?;
                    unsafe { runnable.run(prompt) }
                }
                _ => Err(anyhow!("Unknown configuration").into()),
            }
        }
        // "gpt" => {
        //     let model = load_build(path, gguf)?;
        //     let runnable: Gpt = LLM::build(&model, tokenizer_path)?;
        //     run(runnable, prompt)
        // }
        _ => anyhow::Result::Err(anyhow!("Unsupported architecture"))?,
    }?;
    Ok(())
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    file: String,

    #[arg(short, long)]
    tokenizer: String,

    #[arg(short, long)]
    prompt: String,

    #[arg(short, long, default_value_t = false)]
    clone: bool,

    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    #[arg(long, default_value_t = 0.7)]
    temp: f32,

    #[arg(short, long)]
    seed: Option<u64>,
}

fn main() {
    let args = Args::parse();

    let mut rng = match args.seed {
        Some(n) => StdRng::seed_from_u64(n),
        None => StdRng::from_entropy(),
    };

    let _r: f32 = rng.gen_range(0.0..1.0);

    let result = process(&args.file, &args.tokenizer, &args.prompt, args.clone);

    match result {
        Err(e) => {
            println!("tinyllama: error: {}", e)
        }
        _ => println!("tinyllama: done"),
    }
}
