use crate::model::LLM;
use anyhow::anyhow;
use num_traits::float::Float;
use tokenizers::tokenizer::Tokenizer;
use yloader::*;
use ymath::function::{acc, cp, matmul, rmsnorm, softmax};
use ymath::tensor::*;

type DefaultMmap = MmapStore<f32, f32>;

#[derive(Debug, Clone, Copy)]
struct LlamaParams<U, T = usize> {
    block_count: T,
    _context_length: T,
    embedding_length: T,
    feed_forward_length: T,
    attention_head_count: T,
    attention_head_count_kv: T,
    attention_layer_norm_rms_epsilon: U,
    rope_freq_base: U,
    _rope_dimension_count: T,
    vocab_size: T,
    max_seq_len: T, // This could be context_length or lower to save space
    attention_kv_length: T,
}

#[allow(dead_code)]
pub struct Llama<
    'a,
    T: Float,
    TokenEmbd = DefaultMmap,
    Output = DefaultMmap,
    OutputNorm = DefaultMmap,
    AttnK = DefaultMmap,
    AttnQ = DefaultMmap,
    AttnV = DefaultMmap,
    AttnNorm = DefaultMmap,
    FfnDown = DefaultMmap,
    FfnGate = DefaultMmap,
    FfnNorm = DefaultMmap,
    FfnUp = DefaultMmap,
    AttnOutput = DefaultMmap,
> where
    AttnQ: TensorTypes<T, 2>,
    AttnK: TensorTypes<T, 2>,
    AttnV: TensorTypes<T, 2>,
    AttnNorm: TensorTypes<T, 1>,
    AttnOutput: TensorTypes<T, 2>,
    FfnUp: TensorTypes<T, 2>,
    FfnDown: TensorTypes<T, 2>,
    FfnNorm: TensorTypes<T, 1>,
    FfnGate: TensorTypes<T, 2>,
    TokenEmbd: TensorTypes<T, 2>,
    Output: TensorTypes<T, 2>,
    OutputNorm: TensorTypes<T, 1>,
{
    params: LlamaParams<T, usize>,
    blocks: Vec<
        LlamaBlock<
            'a,
            T,
            AttnK,
            AttnQ,
            AttnV,
            AttnNorm,
            FfnDown,
            FfnGate,
            FfnNorm,
            FfnUp,
            AttnOutput,
        >,
    >,
    token_embd: Tensor2<'a, T, TokenEmbd>,
    output: Tensor2<'a, T, Output>,
    output_norm: Vector<'a, T, OutputNorm>,
    tokenizer: Tokenizer,
}

#[allow(dead_code)]
struct LlamaBlock<
    'a,
    T: Float,
    AttnK,
    AttnQ,
    AttnV,
    AttnNorm,
    FfnDown,
    FfnGate,
    FfnNorm,
    FfnUp,
    AttnOutput,
> where
    AttnQ: TensorTypes<T, 2>,
    AttnK: TensorTypes<T, 2>,
    AttnV: TensorTypes<T, 2>,
    AttnNorm: TensorTypes<T, 1>,
    AttnOutput: TensorTypes<T, 2>,
    FfnUp: TensorTypes<T, 2>,
    FfnDown: TensorTypes<T, 2>,
    FfnNorm: TensorTypes<T, 1>,
    FfnGate: TensorTypes<T, 2>,
{
    i: usize,
    params: LlamaParams<T>,

    // Model weights
    attn_q: Tensor2<'a, T, AttnQ>,
    attn_k: Tensor2<'a, T, AttnK>,
    attn_v: Tensor2<'a, T, AttnV>,
    attn_norm: Vector<'a, T, AttnNorm>,
    attn_ouput: Tensor2<'a, T, AttnOutput>,
    ffn_down: Tensor2<'a, T, FfnDown>,
    ffn_up: Tensor2<'a, T, FfnUp>,
    ffn_norm: Vector<'a, T, FfnNorm>,
    ffn_gate: Tensor2<'a, T, FfnGate>,

    // Block state
    xb: VectorMut<'a, T>,
    xb2: VectorMut<'a, T>,
    hb: VectorMut<'a, T>,
    hb2: VectorMut<'a, T>,
    q: VectorMut<'a, T>,
    k_cache: Tensor2Mut<'a, T>,
    v_cache: Tensor2Mut<'a, T>,
    attn_score: Tensor2Mut<'a, T>,
}

impl<'a, T: Float, AttnK, AttnQ, AttnV, AttnNorm, FfnDown, FfnGate, FfnNorm, FfnUp, AttnOutput>
    LlamaBlock<'a, T, AttnK, AttnQ, AttnV, AttnNorm, FfnDown, FfnGate, FfnNorm, FfnUp, AttnOutput>
where
    AttnQ: TensorTypes<T, 2>,
    Tensor<'a, T, 2, AttnQ>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, AttnQ, &'a ModelFile>,

    AttnK: TensorTypes<T, 2>,
    Tensor<'a, T, 2, AttnK>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, AttnK, &'a ModelFile>,

    AttnV: TensorTypes<T, 2>,
    Tensor<'a, T, 2, AttnV>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, AttnV, &'a ModelFile>,

    AttnNorm: TensorTypes<T, 1>,
    Tensor<'a, T, 1, AttnNorm>: TReader<T, 1>,
    GGUFTensor<()>: Tensorify<'a, T, 1, AttnNorm, &'a ModelFile>,

    AttnOutput: TensorTypes<T, 2>,
    Tensor<'a, T, 2, AttnOutput>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, AttnOutput, &'a ModelFile>,

    FfnUp: TensorTypes<T, 2>,
    Tensor<'a, T, 2, FfnUp>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, FfnUp, &'a ModelFile>,

    FfnDown: TensorTypes<T, 2>,
    Tensor<'a, T, 2, FfnDown>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, FfnDown, &'a ModelFile>,

    FfnNorm: TensorTypes<T, 1>,
    Tensor<'a, T, 1, FfnNorm>: TReader<T, 1>,
    GGUFTensor<()>: Tensorify<'a, T, 1, FfnNorm, &'a ModelFile>,

    FfnGate: TensorTypes<T, 2>,
    Tensor<'a, T, 2, FfnGate>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, FfnGate, &'a ModelFile>,
{
    fn new(model: &'a ModelFile, i: usize, params: LlamaParams<T>) -> Result<Self, anyhow::Error> {
        macro_rules! build_tensor {
            ($s:expr) => {{
                let name = format!($s, i);
                model
                    .tensors
                    .get(&name)
                    .ok_or_else(|| anyhow!("tensor {} not found", name))
                    .map(|x| x.to_tensor(model))
                    .unwrap()
                    .unwrap()
            }};
        }

        // Attention
        let attn_q = build_tensor!("blk.{}.attn_q.weight");
        let attn_k = build_tensor!("blk.{}.attn_k.weight");
        let attn_v = build_tensor!("blk.{}.attn_v.weight");
        let attn_norm = build_tensor!("blk.{}.attn_norm.weight");
        let attn_ouput = build_tensor!("blk.{}.attn_output.weight");

        // Forward network
        let ffn_down = build_tensor!("blk.{}.ffn_down.weight");
        let ffn_up = build_tensor!("blk.{}.ffn_up.weight");
        let ffn_norm = build_tensor!("blk.{}.ffn_norm.weight");
        let ffn_gate = build_tensor!("blk.{}.ffn_gate.weight");

        // Mutable
        let xb = VectorMut::new(params.embedding_length);
        let xb2 = VectorMut::new(params.embedding_length);
        let hb = VectorMut::new(params.feed_forward_length); // Hidden dim
        let hb2 = VectorMut::new(params.feed_forward_length); // Hidden dim
        let q = VectorMut::new(params.embedding_length);
        let k_cache = Tensor2Mut::new(params.max_seq_len, params.attention_kv_length);
        let v_cache = Tensor2Mut::new(params.max_seq_len, params.attention_kv_length);
        let attn_score = Tensor2Mut::new(params.attention_head_count, params.max_seq_len);

        Ok(LlamaBlock {
            i,
            params,
            attn_q,
            attn_k,
            attn_v,
            attn_norm,
            attn_ouput,
            ffn_down,
            ffn_gate,
            ffn_norm,
            ffn_up,
            xb,
            xb2,
            hb,
            hb2,
            q,
            k_cache,
            v_cache,
            attn_score,
        })
    }

    #[inline(always)]
    unsafe fn forward(&mut self, x: &mut VectorMut<T>, pos: usize) {
        let xb = &mut self.xb;
        let xb2 = &mut self.xb2;
        let hb = &mut self.hb;
        let hb2 = &mut self.hb2;

        rmsnorm(
            xb,
            x,
            &mut self.attn_norm,
            self.params.attention_layer_norm_rms_epsilon,
        );

        // q, v and k for this position
        let q = &mut self.q;
        let k = &mut self.k_cache.row(pos);
        let v = &mut self.v_cache.row(pos);
        matmul::<T>(q, &mut self.attn_q, xb);
        matmul::<T>(k, &mut self.attn_k, xb);
        matmul::<T>(v, &mut self.attn_v, xb);

        // RoPE
        let attn_head_size = self.params.embedding_length / self.params.attention_head_count;
        let kv_mul = 4; // TODO: remove hardcoded
        debug_assert!(attn_head_size == 128);
        for i in 0..self.params.attention_head_count {
            for j in (0..attn_head_size).step_by(2) {
                let freq = T::one()
                    / T::powf(
                        self.params.rope_freq_base,
                        T::from(j).unwrap() / T::from(attn_head_size).unwrap(),
                    );
                let val = T::from(pos).unwrap() * freq;
                let fcr = T::cos(val);
                let fci = T::sin(val);
                let q0 = q[i * attn_head_size + j];
                let q1 = q[i * attn_head_size + j + 1];
                q[i * attn_head_size + j] = q0 * fcr - q1 * fci;
                q[i * attn_head_size + j + 1] = q0 * fci + q1 * fcr;
                if i < self.params.attention_head_count_kv {
                    let k0 = k[i * attn_head_size + j];
                    let k1 = k[i * attn_head_size + j + 1];
                    k[i * attn_head_size + j] = k0 * fcr - k1 * fci;
                    k[i * attn_head_size + j + 1] = k0 * fci + k1 * fcr;
                }
            }
        }

        // Multihead attention
        for h in 0..self.params.attention_head_count {
            let attn_score = &mut self.attn_score.row(h);
            let q_offset = h * attn_head_size;
            for t in 0..(pos + 1) {
                let mut score = T::zero();
                let k = &mut self.k_cache.row(t);
                let k_offset = (h / kv_mul) * attn_head_size;
                for i in 0..attn_head_size {
                    score = score + q[q_offset + i] * k[k_offset + i];
                }
                score = score / T::sqrt(T::from(attn_head_size).unwrap());
                attn_score[t] = score;
            }

            // Softmax from 0..p inclusively
            softmax(attn_score, pos + 1);

            // Weighted sum of the values, store back into xb
            let xb_offset = h * attn_head_size;
            for i in 0..attn_head_size {
                xb[xb_offset + i] = T::zero(); // TODO: Optimize? Can we memset?
            }
            for t in 0..(pos + 1) {
                // Attention weight for this timesetp
                let a = attn_score[t];
                let v = &mut self.v_cache.row(t);
                let v_offset = (h / kv_mul) * attn_head_size;
                // Weighted value
                for i in 0..attn_head_size {
                    xb[xb_offset + i] = xb[xb_offset + i] + a * v[v_offset + i];
                }
            }
        }

        // Output of attention
        matmul::<T>(xb2, &mut self.attn_ouput, xb);

        // Residual
        acc(x, xb2);

        // Ffn rmsnorm
        rmsnorm(
            xb,
            x,
            &mut self.ffn_norm,
            self.params.attention_layer_norm_rms_epsilon,
        );

        // Non-linearity
        matmul::<T>(hb, &mut self.ffn_gate, xb);
        matmul::<T>(hb2, &mut self.ffn_up, xb);
        for i in 0..self.params.feed_forward_length {
            let mut val = hb[i];
            val = val * (T::one() / (T::one() + T::exp(-val)));
            val = val * hb2[i];
            hb[i] = val;
        }

        // Ffn output
        matmul::<T>(xb, &mut self.ffn_down, hb);

        // Residual
        acc(x, xb);
    }
}

fn conv_err(b: Box<dyn std::error::Error + Send + Sync>) -> Box<dyn std::error::Error> {
    b
}

impl<
        'a,
        T: Float,
        TokenEmbd,
        Output,
        OutputNorm,
        AttnK,
        AttnQ,
        AttnV,
        AttnNorm,
        FfnDown,
        FfnGate,
        FfnNorm,
        FfnUp,
        AttnOutput,
    >
    Llama<
        'a,
        T,
        TokenEmbd,
        Output,
        OutputNorm,
        AttnK,
        AttnQ,
        AttnV,
        AttnNorm,
        FfnDown,
        FfnGate,
        FfnNorm,
        FfnUp,
        AttnOutput,
    >
where
    AttnQ: TensorTypes<T, 2>,
    Tensor<'a, T, 2, AttnQ>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, AttnQ, &'a ModelFile>,

    AttnK: TensorTypes<T, 2>,
    Tensor<'a, T, 2, AttnK>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, AttnK, &'a ModelFile>,

    AttnV: TensorTypes<T, 2>,
    Tensor<'a, T, 2, AttnV>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, AttnV, &'a ModelFile>,

    AttnNorm: TensorTypes<T, 1>,
    Tensor<'a, T, 1, AttnNorm>: TReader<T, 1>,
    GGUFTensor<()>: Tensorify<'a, T, 1, AttnNorm, &'a ModelFile>,

    AttnOutput: TensorTypes<T, 2>,
    Tensor<'a, T, 2, AttnOutput>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, AttnOutput, &'a ModelFile>,

    FfnUp: TensorTypes<T, 2>,
    Tensor<'a, T, 2, FfnUp>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, FfnUp, &'a ModelFile>,

    FfnDown: TensorTypes<T, 2>,
    Tensor<'a, T, 2, FfnDown>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, FfnDown, &'a ModelFile>,

    FfnNorm: TensorTypes<T, 1>,
    Tensor<'a, T, 1, FfnNorm>: TReader<T, 1>,
    GGUFTensor<()>: Tensorify<'a, T, 1, FfnNorm, &'a ModelFile>,

    FfnGate: TensorTypes<T, 2>,
    Tensor<'a, T, 2, FfnGate>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, FfnGate, &'a ModelFile>,

    TokenEmbd: TensorTypes<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, TokenEmbd, &'a ModelFile>,

    Output: TensorTypes<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, Output, &'a ModelFile>,

    OutputNorm: TensorTypes<T, 1>,
    GGUFTensor<()>: Tensorify<'a, T, 1, OutputNorm, &'a ModelFile>,
{
    fn new(model: &'a ModelFile, tokenizer_path: &str) -> Result<Self, anyhow::Error> {
        let header = &model.header;

        // Huggingface tokenizer
        // TODO: implement a local algorithm using using data in GGUF
        let tokenizer = match Tokenizer::from_file(tokenizer_path) {
            Ok(tokenizer) => Ok(tokenizer),
            Err(err) => Err(anyhow!("could not create tokenizer: {}", err.to_string())),
        }?;

        let embedding_length = header_find_usize(header, "llama.embedding_length")?;
        let attention_head_count_kv = header_find_usize(header, "llama.attention.head_count_kv")?;
        let attention_head_count = header_find_usize(header, "llama.attention.head_count")?;
        let params: LlamaParams<T> = LlamaParams {
            block_count: header_find_usize(header, "llama.block_count")?,
            _context_length: header_find_usize(header, "llama.context_length")?,
            embedding_length,
            feed_forward_length: header_find_usize(header, "llama.feed_forward_length")?,
            attention_head_count,
            attention_head_count_kv,
            attention_layer_norm_rms_epsilon: T::from(header_find_f32(
                header,
                "llama.attention.layer_norm_rms_epsilon",
            )?)
            .unwrap(),
            rope_freq_base: T::from(header_find_f32(header, "llama.rope.freq_base")?).unwrap(),
            _rope_dimension_count: header_find_usize(header, "llama.rope.dimension_count")?,
            vocab_size: header_find_usize(header, "llama.vocab_size")?,
            max_seq_len: header_find_usize(header, "llama.context_length")?,
            attention_kv_length: embedding_length * attention_head_count_kv / attention_head_count,
        };

        macro_rules! build_tensor {
            ($s:expr) => {{
                let name = $s;
                model
                    .tensors
                    .get(name)
                    .ok_or_else(|| anyhow!("tensor {} not found", name))
                    .map(|x| x.to_tensor(model))?
                    .unwrap()
            }};
        }

        let token_embd = build_tensor!("token_embd.weight");
        let output = build_tensor!("output.weight");
        let output_norm = build_tensor!("output_norm.weight");

        let blocks: Result<Vec<_>, anyhow::Error> = (0..params.block_count)
            .into_iter()
            .map(|i| LlamaBlock::new(&model, i, params))
            .collect();

        let blocks = blocks?;

        Ok(Llama {
            params,
            blocks,
            token_embd,
            output,
            output_norm,
            // tokens,
            tokenizer,
        })
    }
}

impl<
        'a,
        T: Float,
        TokenEmbd,
        Output,
        OutputNorm,
        AttnK,
        AttnQ,
        AttnV,
        AttnNorm,
        FfnDown,
        FfnGate,
        FfnNorm,
        FfnUp,
        AttnOutput,
    > LLM<'a, T, u32, ModelFile>
    for Llama<
        'a,
        T,
        TokenEmbd,
        Output,
        OutputNorm,
        AttnK,
        AttnQ,
        AttnV,
        AttnNorm,
        FfnDown,
        FfnGate,
        FfnNorm,
        FfnUp,
        AttnOutput,
    >
where
    AttnQ: TensorTypes<T, 2>,
    Tensor<'a, T, 2, AttnQ>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, AttnQ, &'a ModelFile>,

    AttnK: TensorTypes<T, 2>,
    Tensor<'a, T, 2, AttnK>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, AttnK, &'a ModelFile>,

    AttnV: TensorTypes<T, 2>,
    Tensor<'a, T, 2, AttnV>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, AttnV, &'a ModelFile>,

    AttnNorm: TensorTypes<T, 1>,
    Tensor<'a, T, 1, AttnNorm>: TReader<T, 1>,
    GGUFTensor<()>: Tensorify<'a, T, 1, AttnNorm, &'a ModelFile>,

    AttnOutput: TensorTypes<T, 2>,
    Tensor<'a, T, 2, AttnOutput>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, AttnOutput, &'a ModelFile>,

    FfnUp: TensorTypes<T, 2>,
    Tensor<'a, T, 2, FfnUp>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, FfnUp, &'a ModelFile>,

    FfnDown: TensorTypes<T, 2>,
    Tensor<'a, T, 2, FfnDown>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, FfnDown, &'a ModelFile>,

    FfnNorm: TensorTypes<T, 1>,
    Tensor<'a, T, 1, FfnNorm>: TReader<T, 1>,
    GGUFTensor<()>: Tensorify<'a, T, 1, FfnNorm, &'a ModelFile>,

    FfnGate: TensorTypes<T, 2>,
    Tensor<'a, T, 2, FfnGate>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, FfnGate, &'a ModelFile>,

    TokenEmbd: TensorTypes<T, 2> + TensorTypes<T, 1>,
    Tensor<'a, T, 2, TokenEmbd>: TReader<T, 2> + Rowable<'a, T, TokenEmbd>,
    Tensor<'a, T, 1, TokenEmbd>: TReader<T, 1>,
    GGUFTensor<()>: Tensorify<'a, T, 2, TokenEmbd, &'a ModelFile>,

    Output: TensorTypes<T, 2>,
    Tensor<'a, T, 2, Output>: TReader<T, 2>,
    GGUFTensor<()>: Tensorify<'a, T, 2, Output, &'a ModelFile>,

    OutputNorm: TensorTypes<T, 1>,
    Tensor<'a, T, 1, OutputNorm>: TReader<T, 1>,
    GGUFTensor<()>: Tensorify<'a, T, 1, OutputNorm, &'a ModelFile>,
{
    fn build<'b>(model: &'a ModelFile, tokenizer_path: &str) -> Result<Self, anyhow::Error> {
        Ok(Llama::new(&model, tokenizer_path)?)
    }

    fn embedding_length(&self) -> usize {
        self.params.embedding_length
    }
    fn vocab_size(&self) -> usize {
        self.params.vocab_size
    }
    fn block_count(&self) -> usize {
        self.params.block_count
    }

    fn encode(&self, input: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let r = self.tokenizer.encode(input, false).map_err(conv_err)?;
        Ok(r.get_ids().iter().map(|t| *t).collect())
    }

    fn embed(&mut self, x: &mut VectorMut<T>, token: u32, _pos: usize) {
        cp(x, &mut self.token_embd.row(token as usize))
    }

    fn decode(&self, tokens: &Vec<u32>) -> String {
        self.tokenizer.decode(tokens, false).unwrap()
    }

    unsafe fn forward(&mut self, x: &mut VectorMut<T>, pos: usize) {
        self.blocks
            .iter_mut()
            .for_each(|block| block.forward(x, pos));
    }

    unsafe fn block_forward(&mut self, x: &mut VectorMut<T>, pos: usize, block: usize) {
        self.blocks[block].forward(x, pos)
    }

    unsafe fn logits(&mut self, logits: &mut VectorMut<T>, x: &mut VectorMut<T>) {
        // Final rmsnorm
        let mut x2: TensorMut<T, 1> = VectorMut::new(self.params.embedding_length);
        rmsnorm(
            &mut x2,
            x,
            &mut self.output_norm,
            self.params.attention_layer_norm_rms_epsilon,
        );
        // Last act: logits
        matmul::<T>(logits, &mut self.output, &mut x2);
    }
}

pub fn llama_find_type(model: &ModelFile) -> Result<&str, anyhow::Error> {
    let find = |name| match model.tensors.get(name) {
        Some(t) => Ok(t.tensor_type),
        None => Err(anyhow!("could not find tensor")),
    };

    let token_embd = find("token_embd.weight")?;
    let output = find("output.weight")?;
    println!("{:?}", output);
    if token_embd == GGMLType::F32 {
        Ok("F32")
    } else {
        Ok("F16")
    }
}
