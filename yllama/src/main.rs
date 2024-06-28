pub mod gpt;
pub mod llama;
pub use gpt::Gpt;
pub mod llm;

use std::str;
use anyhow::anyhow;
use clap::Parser;
use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use llm::LLM;
use llama::Llama;
use yloader::{load_build, load_fast};
use ymath::tensor::{MmapStore, VecStore};

unsafe fn process(
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
            type B = MmapStore<f32, f16>;
            type C = VecStore<f32, f16>;
            type D = VecStore<f32, f32>;
            let typ = llama::llama_find_type(&model)?;
            // This is UGLY - TODO: improve on it!
            match typ {
                "F16" => {
                    type LlamaType<'a> = Llama<'a, f32, C, B, A, B, B, B, D, B, B, A, B, B>;
                    let mut runnable: LlamaType =
                        LLM::build(&model, tokenizer_path)?;
                    runnable.run(prompt)
                }
                "F32" => {
                    let mut runnable: Llama<f32> = LLM::build(&model, tokenizer_path)?;
                    runnable.run(prompt)
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

    let result = unsafe { process(&args.file, &args.tokenizer, &args.prompt, args.clone) };

    match result {
        Err(e) => {
            println!("tinyllama: error: {}", e)
        }
        _ => println!("tinyllama: done"),
    }
}
