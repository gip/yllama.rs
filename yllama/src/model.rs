use ymath::tensor::VectorMut;

pub trait LLM<'a, T, TK, M>: Sized {
    fn build(model: &'a M, tokenizer_path: &str) -> Result<Self, anyhow::Error>
    where
        Self: Sized;

    fn embedding_length(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn block_count(&self) -> usize;

    fn encode(&self, input: &str) -> Result<Vec<TK>, Box<dyn std::error::Error>>;
    fn embed(&mut self, x: &mut VectorMut<T>, token: TK, pos: usize);
    unsafe fn forward(&mut self, x: &mut VectorMut<T>, pos: usize);
    unsafe fn logits(&mut self, logits: &mut VectorMut<T>, x: &mut VectorMut<T>);
    fn decode(&self, token: &Vec<TK>) -> String;

    unsafe fn block_forward(&mut self, x: &mut VectorMut<T>, pos: usize, block: usize);
}
