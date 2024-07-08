use num_traits::float::Float;
use std::convert::TryInto;
use std::fmt::Debug;
use ymath::function::max;
use ymath::tensor::VectorMut;

pub trait Instantiable<TA, CTX> {
    fn instantiate(_: CTX) -> Result<Self, anyhow::Error>
    where
        Self: Sized;
}

pub trait LLM<'a, TA, T: Float, TK: Copy, M, const EMBED: usize, const VOCAB: usize>:
    Sized
where
    usize: TryInto<TK>,
    <usize as TryInto<TK>>::Error: Debug,
{
    fn block_count(&self) -> usize;

    fn encode(&self, input: &str) -> Result<Vec<TK>, Box<dyn std::error::Error>>;
    fn embed(&mut self, x: &mut VectorMut<T, EMBED>, token: TK, pos: usize);
    unsafe fn forward(&mut self, x: &mut VectorMut<T, EMBED>, pos: usize);
    unsafe fn logits(&mut self, logits: &mut VectorMut<T, VOCAB>, x: &mut VectorMut<T, EMBED>);
    fn decode(&self, token: &Vec<TK>) -> String;

    unsafe fn block_forward(&mut self, x: &mut VectorMut<T, EMBED>, pos: usize, block: usize);

    unsafe fn run(&mut self, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
        let input = prompt;
        let tokens: Vec<TK> = self.encode(input)?;
        let mut logits: VectorMut<T, VOCAB> = VectorMut::new_vector();
        let mut x: VectorMut<T, EMBED> = VectorMut::new_vector();
        let mut next_token = tokens[0];
        let mut chat = vec![];
        for pos in 0..1024 {
            chat.push(next_token);
            println!("{}", self.decode(&chat));
            self.embed(&mut x, next_token, pos);
            self.forward(&mut x, pos);
            self.logits(&mut logits, &mut x);
            let (tk, _) = max(&mut logits);
            if pos + 1 < tokens.len() {
                next_token = tokens[pos + 1];
            } else {
                next_token = tk.try_into().unwrap();
            }
        }
        Ok(())
    }
}
