pub mod gpt;
pub mod llama;
pub mod model;

use anyhow::anyhow;
use clap::Parser;
pub use gpt::Gpt;
use half::f16;
use llama::Llama;
use model::LLM;
use num_traits::float::Float;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::str;
use yloader::{load_build, load_fast};
use ymath::{max, MmapStore, VectorMut};

unsafe fn run<'a, T: Float, M>(
    mut llm: impl LLM<'a, T, u32, M> + 'a,
    prompt: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = prompt;
    let tokens: Vec<u32> = llm.encode(input)?;
    let embed_size = llm.embedding_length();
    let vocab_size = llm.vocab_size();
    let mut logits = VectorMut::new(vocab_size);
    let mut x: VectorMut<T> = VectorMut::new(embed_size);
    let mut next_token = tokens[0];
    let mut chat = vec![];
    for pos in 0..1024 {
        chat.push(next_token);
        println!("{}", llm.decode(&chat));
        llm.embed(&mut x, next_token, pos);
        llm.forward(&mut x, pos);
        llm.logits(&mut logits, &mut x);
        let (tk, _) = max(&mut logits);
        if pos + 1 < tokens.len() {
            next_token = tokens[pos + 1];
        } else {
            next_token = tk as u32;
        }
    }
    Ok(())
}

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
            type A = MmapStore<f32>;
            type B = MmapStore<f16>;
            let typ = llama::llama_find_type(&model)?;
            match typ {
                "F16" => {
                    let runnable: Llama<f32, B, B, A, B, B, B, A, B, B, A, B, B> =
                    LLM::build(&model, tokenizer_path)?;
                    run(runnable, prompt)
                },
                "F32" => {
                    let runnable: Llama<f32> =
                    LLM::build(&model, tokenizer_path)?;
                    run(runnable, prompt)
                },
                _ => Err(anyhow!("Unknown configuration").into())
            }
        }
        "gpt" => {
            let model = load_build(path, gguf)?;
            let runnable: Gpt = LLM::build(&model, tokenizer_path)?;
            run(runnable, prompt)
        }
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
