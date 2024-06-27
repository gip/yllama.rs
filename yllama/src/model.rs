use ymath::tensor::VectorMut;

pub trait LLM<'a, T, TK, M, const EMBED: usize, const VOCAB: usize>: Sized {
    fn build(model: &'a M, tokenizer_path: &str) -> Result<Self, anyhow::Error>
    where
        Self: Sized;

    fn block_count(&self) -> usize;

    fn encode(&self, input: &str) -> Result<Vec<TK>, Box<dyn std::error::Error>>;
    fn embed(&mut self, x: &mut VectorMut<T, EMBED>, token: TK, pos: usize);
    unsafe fn forward(&mut self, x: &mut VectorMut<T, EMBED>, pos: usize);
    unsafe fn logits(&mut self, logits: &mut VectorMut<T, VOCAB>, x: &mut VectorMut<T, EMBED>);
    fn decode(&self, token: &Vec<TK>) -> String;

    unsafe fn block_forward(&mut self, x: &mut VectorMut<T, EMBED>, pos: usize, block: usize);
}
