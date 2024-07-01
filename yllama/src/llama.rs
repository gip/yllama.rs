use crate::llm::LLM;
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
    _max_seq_len: T,
    _attention_kv_length: T,
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
    const EMBED: usize = 4096,
    const VOCAB: usize = 128256,
    const FF: usize = 14336,
    const KV: usize = 1024,
    const CONTEXT: usize = 2048,
> where
    AttnQ: TensorTypes<T, M<EMBED, EMBED>>,
    AttnK: TensorTypes<T, M<EMBED, KV>>,
    AttnV: TensorTypes<T, M<EMBED, KV>>,
    AttnNorm: TensorTypes<T, V<EMBED>>,
    AttnOutput: TensorTypes<T, M<EMBED, EMBED>>,
    FfnUp: TensorTypes<T, M<EMBED, FF>>,
    FfnDown: TensorTypes<T, M<FF, EMBED>>,
    FfnNorm: TensorTypes<T, V<EMBED>>,
    FfnGate: TensorTypes<T, M<EMBED, FF>>,
    TokenEmbd: TensorTypes<T, M<EMBED, VOCAB>>,
    Output: TensorTypes<T, M<EMBED, VOCAB>>,
    OutputNorm: TensorTypes<T, V<EMBED>>,
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
            EMBED,
            VOCAB,
            FF,
            KV,
            CONTEXT,
        >,
    >,
    token_embd: Tensor2Imm<'a, T, EMBED, VOCAB, TokenEmbd>,
    output: Tensor2Imm<'a, T, EMBED, VOCAB, Output>,
    output_norm: VectorImm<'a, T, EMBED, OutputNorm>,
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
    const EMBED: usize,
    const VOCAB: usize,
    const FF: usize,
    const KV: usize,
    const CONTEXT: usize,
> where
    AttnQ: TensorTypes<T, M<EMBED, EMBED>>,
    AttnK: TensorTypes<T, M<EMBED, KV>>,
    AttnV: TensorTypes<T, M<EMBED, KV>>,
    AttnNorm: TensorTypes<T, V<EMBED>>,
    AttnOutput: TensorTypes<T, M<EMBED, EMBED>>,
    FfnUp: TensorTypes<T, M<EMBED, FF>>,
    FfnDown: TensorTypes<T, M<FF, EMBED>>,
    FfnNorm: TensorTypes<T, V<EMBED>>,
    FfnGate: TensorTypes<T, M<EMBED, FF>>,
{
    i: usize,
    params: LlamaParams<T>,

    // Model weights
    attn_q: Tensor2Imm<'a, T, EMBED, EMBED, AttnQ>,
    attn_k: Tensor2Imm<'a, T, EMBED, KV, AttnK>,
    attn_v: Tensor2Imm<'a, T, EMBED, KV, AttnV>,
    attn_norm: VectorImm<'a, T, EMBED, AttnNorm>,
    attn_ouput: Tensor2Imm<'a, T, EMBED, EMBED, AttnOutput>,
    ffn_down: Tensor2Imm<'a, T, FF, EMBED, FfnDown>,
    ffn_up: Tensor2Imm<'a, T, EMBED, FF, FfnUp>,
    ffn_norm: VectorImm<'a, T, EMBED, FfnNorm>,
    ffn_gate: Tensor2Imm<'a, T, EMBED, FF, FfnGate>,

    // Block state
    xb: VectorMut<'a, T, EMBED>,
    xb2: VectorMut<'a, T, EMBED>,
    hb: VectorMut<'a, T, FF>,
    hb2: VectorMut<'a, T, FF>,
    q: VectorMut<'a, T, EMBED>,
    k_cache: Tensor2Mut<'a, T, KV, EMBED>,
    v_cache: Tensor2Mut<'a, T, KV, EMBED>,
    attn_score: Tensor2Mut<'a, T, CONTEXT, KV>,
}

impl<
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
        const EMBED: usize,
        const VOCAB: usize,
        const FF: usize,
        const KV: usize,
        const CONTEXT: usize,
    >
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
        EMBED,
        VOCAB,
        FF,
        KV,
        CONTEXT,
    >
where
    AttnQ: TensorTypes<T, M<EMBED, EMBED>>,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnQ>: TReader<T, M<EMBED, EMBED>> + 'a,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, EMBED>, AttnQ, &'a ModelFile>,

    AttnK: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnK>: TReader<T, M<EMBED, KV>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, KV>, AttnK, &'a ModelFile>,

    AttnV: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnV>: TReader<T, M<EMBED, KV>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, KV>, AttnV, &'a ModelFile>,

    AttnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, AttnNorm>: TReader<T, V<EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, V<EMBED>, AttnNorm, &'a ModelFile>,

    AttnOutput: TensorTypes<T, M<EMBED, EMBED>>,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnOutput>: TReader<T, M<EMBED, EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, EMBED>, AttnOutput, &'a ModelFile>,

    FfnUp: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnUp>: TReader<T, M<EMBED, FF>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, FF>, FfnUp, &'a ModelFile>,

    FfnDown: TensorTypes<T, M<FF, EMBED>>,
    Tensor<'a, false, T, M<FF, EMBED>, FfnDown>: TReader<T, M<FF, EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, M<FF, EMBED>, FfnDown, &'a ModelFile>,

    FfnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, FfnNorm>: TReader<T, V<EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, V<EMBED>, FfnNorm, &'a ModelFile>,

    FfnGate: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnGate>: TReader<T, M<EMBED, FF>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, FF>, FfnGate, &'a ModelFile>,
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
        let xb = VectorMut::new();
        let xb2 = VectorMut::new();
        let hb = VectorMut::new();
        let hb2 = VectorMut::new();
        let q = VectorMut::new();
        let k_cache = Tensor2Mut::new();
        let v_cache = Tensor2Mut::new();
        let attn_score = Tensor2Mut::new();

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
    unsafe fn forward(&mut self, x: &mut VectorMut<T, EMBED>, pos: usize) {
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
        matmul(q, &mut self.attn_q, xb);
        matmul(k, &mut self.attn_k, xb);
        matmul(v, &mut self.attn_v, xb);

        let mut qw = q.writer();
        let mut kw = k.writer();

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
                let q0 = qw.get(i * attn_head_size + j);
                let q1 = qw.get(i * attn_head_size + j + 1);
                qw.set(i * attn_head_size + j, q0 * fcr - q1 * fci);
                qw.set(i * attn_head_size + j + 1, q0 * fci + q1 * fcr);
                if i < self.params.attention_head_count_kv {
                    let k0 = kw.get(i * attn_head_size + j);
                    let k1 = kw.get(i * attn_head_size + j + 1);
                    kw.set(i * attn_head_size + j, k0 * fcr - k1 * fci);
                    kw.set(i * attn_head_size + j + 1, k0 * fci + k1 * fcr);
                }
            }
        }

        // Multihead attention
        for h in 0..self.params.attention_head_count {
            let attn_score = &mut self.attn_score.row(h);
            {
                let mut attn_score_w = attn_score.writer();
                let q_offset = h * attn_head_size;
                for t in 0..(pos + 1) {
                    let mut score = T::zero();
                    let k = &mut self.k_cache.row(t);
                    let kw = k.writer();
                    let k_offset = (h / kv_mul) * attn_head_size;
                    for i in 0..attn_head_size {
                        score = score + qw.get(q_offset + i) * kw.get(k_offset + i);
                    }
                    score = score / T::sqrt(T::from(attn_head_size).unwrap());
                    attn_score_w.set(t, score);
                }
            }

            // Softmax from 0..p inclusively
            softmax(attn_score, pos + 1);

            // Weighted sum of the values, store back into xb
            {
                let xb_offset = h * attn_head_size;
                let mut xbw = xb.writer();
                let attn_score_w = attn_score.writer();
                for i in 0..attn_head_size {
                    xbw.set(xb_offset + i, T::zero()); // TODO: Optimize? Can we memset?
                }
                for t in 0..(pos + 1) {
                    // Attention weight for this timesetp
                    let a = attn_score_w.get(t);
                    let v = &mut self.v_cache.row(t);
                    let v_w = v.writer();
                    let v_offset = (h / kv_mul) * attn_head_size;
                    // Weighted value
                    for i in 0..attn_head_size {
                        xbw.set(xb_offset + i, xbw.get(xb_offset + i) + a * v_w.get(v_offset + i));
                    }
                }
            }
        }

        // Output of attention
        matmul(xb2, &mut self.attn_ouput, xb);

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
        matmul(hb, &mut self.ffn_gate, xb);
        matmul(hb2, &mut self.ffn_up, xb);
        {
            let mut hbw = hb.writer();
            let hb2w = hb2.writer();
            for i in 0..self.params.feed_forward_length {
                let mut val = hbw.get(i);
                val = val * (T::one() / (T::one() + T::exp(-val)));
                val = val * hb2w.get(i);
                hbw.set(i, val);
            }
        }

        // Ffn output
        matmul(xb, &mut self.ffn_down, hb);

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
        const EMBED: usize,
        const VOCAB: usize,
        const FF: usize,
        const KV: usize,
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
        EMBED,
        VOCAB,
        FF,
        KV,
    >
where
    AttnQ: TensorTypes<T, M<EMBED, EMBED>> + 'a,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnQ>: TReader<T, M<EMBED, EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, EMBED>, AttnQ, &'a ModelFile>,

    AttnK: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnK>: TReader<T, M<EMBED, KV>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, KV>, AttnK, &'a ModelFile>,

    AttnV: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnV>: TReader<T, M<EMBED, KV>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, KV>, AttnV, &'a ModelFile>,

    AttnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, AttnNorm>: TReader<T, V<EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, V<EMBED>, AttnNorm, &'a ModelFile>,

    AttnOutput: TensorTypes<T, M<EMBED, EMBED>>,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnOutput>: TReader<T, M<EMBED, EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, EMBED>, AttnOutput, &'a ModelFile>,

    FfnUp: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnUp>: TReader<T, M<EMBED, FF>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, FF>, FfnUp, &'a ModelFile>,

    FfnDown: TensorTypes<T, M<FF, EMBED>>,
    Tensor<'a, false, T, M<FF, EMBED>, FfnDown>: TReader<T, M<FF, EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, M<FF, EMBED>, FfnDown, &'a ModelFile>,

    FfnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, FfnNorm>: TReader<T, V<EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, V<EMBED>, FfnNorm, &'a ModelFile>,

    FfnGate: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnGate>: TReader<T, M<EMBED, FF>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, FF>, FfnGate, &'a ModelFile>,

    TokenEmbd: TensorTypes<T, M<EMBED, VOCAB>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, VOCAB>, TokenEmbd, &'a ModelFile>,

    Output: TensorTypes<T, M<EMBED, VOCAB>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, VOCAB>, Output, &'a ModelFile>,

    OutputNorm: TensorTypes<T, V<EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, V<EMBED>, OutputNorm, &'a ModelFile>,
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
            _max_seq_len: header_find_usize(header, "llama.context_length")?,
            _attention_kv_length: embedding_length * attention_head_count_kv / attention_head_count,
        };

        if params.vocab_size != VOCAB {
            return Err(anyhow!("'llama.vocab_size' doesn't match the static value"));
        }

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
        const EMBED: usize,
        const VOCAB: usize,
        const FF: usize,
        const KV: usize,
    > LLM<'a, T, u32, ModelFile, EMBED, VOCAB>
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
        EMBED,
        VOCAB,
        FF,
        KV,
    >
where
    AttnQ: TensorTypes<T, M<EMBED, EMBED>> + 'a,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnQ>: TReader<T, M<EMBED, EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, EMBED>, AttnQ, &'a ModelFile>,

    AttnK: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnK>: TReader<T, M<EMBED, KV>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, KV>, AttnK, &'a ModelFile>,

    AttnV: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnV>: TReader<T, M<EMBED, KV>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, KV>, AttnV, &'a ModelFile>,

    AttnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, AttnNorm>: TReader<T, V<EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, V<EMBED>, AttnNorm, &'a ModelFile>,

    AttnOutput: TensorTypes<T, M<EMBED, EMBED>>,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnOutput>: TReader<T, M<EMBED, EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, EMBED>, AttnOutput, &'a ModelFile>,

    FfnUp: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnUp>: TReader<T, M<EMBED, FF>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, FF>, FfnUp, &'a ModelFile>,

    FfnDown: TensorTypes<T, M<FF, EMBED>>,
    Tensor<'a, false, T, M<FF, EMBED>, FfnDown>: TReader<T, M<FF, EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, M<FF, EMBED>, FfnDown, &'a ModelFile>,

    FfnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, FfnNorm>: TReader<T, V<EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, V<EMBED>, FfnNorm, &'a ModelFile>,

    FfnGate: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnGate>: TReader<T, M<EMBED, FF>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, FF>, FfnGate, &'a ModelFile>,

    TokenEmbd: TensorTypes<T, M<EMBED, VOCAB>> + 'a,
    Tensor<'a, false, T, M<EMBED, VOCAB>, TokenEmbd>:
        TReader<T, M<EMBED, VOCAB>> + Rowable<T, EMBED, VOCAB, TokenEmbd>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, VOCAB>, TokenEmbd, &'a ModelFile>,

    Output: TensorTypes<T, M<EMBED, VOCAB>>,
    Tensor<'a, false, T, M<EMBED, VOCAB>, Output>: TReader<T, M<EMBED, VOCAB>>,
    GGUFTensor<()>: Tensorify<'a, T, M<EMBED, VOCAB>, Output, &'a ModelFile>,

    OutputNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, OutputNorm>: TReader<T, V<EMBED>>,
    GGUFTensor<()>: Tensorify<'a, T, V<EMBED>, OutputNorm, &'a ModelFile>,
{
    fn build<'b>(model: &'a ModelFile, tokenizer_path: &str) -> Result<Self, anyhow::Error> {
        Ok(Llama::new(&model, tokenizer_path)?)
    }

    fn block_count(&self) -> usize {
        self.params.block_count
    }

    fn encode(&self, input: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let r = self.tokenizer.encode(input, false).map_err(conv_err)?;
        Ok(r.get_ids().iter().map(|t| *t).collect())
    }

    fn embed(&mut self, x: &mut VectorMut<T, EMBED>, token: u32, _pos: usize) {
        cp(x, &mut self.token_embd.row(token as usize))
    }

    fn decode(&self, tokens: &Vec<u32>) -> String {
        self.tokenizer.decode(tokens, false).unwrap()
    }

    unsafe fn forward(&mut self, x: &mut VectorMut<T, EMBED>, pos: usize) {
        self.blocks
            .iter_mut()
            .for_each(|block| block.forward(x, pos));
    }

    unsafe fn block_forward(&mut self, x: &mut VectorMut<T, EMBED>, pos: usize, block: usize) {
        self.blocks[block].forward(x, pos)
    }

    unsafe fn logits(&mut self, logits: &mut VectorMut<T, VOCAB>, x: &mut VectorMut<T, EMBED>) {
        // Final rmsnorm
        let mut x2: VectorMut<T, EMBED> = VectorMut::new();
        rmsnorm(
            &mut x2,
            x,
            &mut self.output_norm,
            self.params.attention_layer_norm_rms_epsilon,
        );
        // Last act: logits
        matmul(logits, &mut self.output, &mut x2);
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
