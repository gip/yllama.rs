use crate::llm::{Instantiable, LLM};
use anyhow::*;
use yloader::*;
use ymath::tensor::*;

type ModelDescription<'a> = ModelFile;

pub struct Gpt {}

impl<'a, TA, CTX, const EMBED: usize, const VOCAB: usize>
    LLM<'a, TA, CTX, f32, u32, ModelDescription<'a>, EMBED, VOCAB> for Gpt
where Gpt: Instantiable<TA, CTX>
{
    fn block_count(&self) -> usize {
        unimplemented!()
    }

    fn encode(&self, _input: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        unimplemented!()
    }

    fn embed(&mut self, _x: &mut VectorMut<f32, EMBED>, _token: u32, _pos: usize) {
        unimplemented!()
    }

    fn decode(&self, _tokens: &Vec<u32>) -> String {
        unimplemented!()
    }

    unsafe fn forward(&mut self, _x: &mut VectorMut<f32, EMBED>, _pos: usize) {
        unimplemented!()
    }

    unsafe fn block_forward(&mut self, _x: &mut VectorMut<f32, EMBED>, _pos: usize, _block: usize) {
        unimplemented!()
    }

    unsafe fn logits(
        &mut self,
        _logits: &mut VectorMut<f32, VOCAB>,
        _x: &mut VectorMut<f32, EMBED>,
    ) {
        unimplemented!()
    }
}
