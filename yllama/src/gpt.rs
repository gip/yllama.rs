use crate::model::LLM;
use anyhow::*;
use yloader::*;
use ymath::*;

type ModelDescription<'a> = ModelFile<GGUFFile<MemLayout<'a, f32>>>;

pub struct Gpt {}

impl<'a> LLM<'a, f32, u32, ModelDescription<'a>> for Gpt {
    fn build<'b>(
        _model: &'a ModelDescription,
        _tokenizer_path: &str,
        _clone: bool
    ) -> Result<Self, anyhow::Error> {
        unimplemented!()
    }

    fn embedding_length(&self) -> usize {
        unimplemented!()
    }
    fn vocab_size(&self) -> usize {
        unimplemented!()
    }
    fn block_count(&self) -> usize {
        unimplemented!()
    }

    fn encode(&self, _input: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        unimplemented!()
    }

    fn embed(&self, _x: &mut VectorMut<f32>, _token: u32, _pos: usize) {
        unimplemented!()
    }

    fn decode(&self, _tokens: &Vec<u32>) -> String {
        unimplemented!()
    }

    unsafe fn forward(&mut self, _x: &mut VectorMut<f32>, _pos: usize) {
        unimplemented!()
    }

    unsafe fn block_forward(&mut self, _x: &mut VectorMut<f32>, _pos: usize, _block: usize) {
        unimplemented!()
    }

    unsafe fn logits(&self, _logits: &mut VectorMut<f32>, _x: &VectorMut<f32>) {
        unimplemented!()
    }
}
