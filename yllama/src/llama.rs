use std::marker::PhantomData;

use crate::llm::{Instantiable, LLM};
use anyhow::anyhow;
use num_traits::float::Float;
use tokenizers::tokenizer::Tokenizer;
use yloader::*;
use ymath::function::{acc, cp, matmul, rmsnorm, softmax};
use ymath::tensor::*;

#[derive(Debug, Clone, Copy)]
pub struct LlamaParams<U, T = usize> {
    pub block_count: T,
    pub _context_length: T,
    pub embedding_length: T,
    pub feed_forward_length: T,
    pub attention_head_count: T,
    pub attention_head_count_kv: T,
    pub attention_layer_norm_rms_epsilon: U,
    pub rope_freq_base: U,
    pub _rope_dimension_count: T,
    pub vocab_size: T,
    pub _max_seq_len: T,
    pub _attention_kv_length: T,
}

#[allow(dead_code)]
pub struct Llama<
    'a,
    TA,
    D,
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
    const CONTEXT: usize,
    Xb = VecStore<T>,
    Xb2 = VecStore<T>,
    Hb = VecStore<T>,
    Hb2 = VecStore<T>,
    Q = VecStore<T>,
    KCache = VecStore<T>,
    VCache = VecStore<T>,
    AttnScore = VecStore<T>,
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
    Xb: TensorTypes<T, V<EMBED>>,
    Xb2: TensorTypes<T, V<EMBED>>,
    Hb: TensorTypes<T, V<FF>>,
    Hb2: TensorTypes<T, V<FF>>,
    Q: TensorTypes<T, V<EMBED>>,
    KCache: TensorTypes<T, M<KV, EMBED>>,
    VCache: TensorTypes<T, M<KV, EMBED>>,
    AttnScore: TensorTypes<T, M<CONTEXT, KV>>,
{
    params: LlamaParams<T, usize>,
    blocks: Vec<
        LlamaBlock<
            'a,
            TA,
            D,
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
            Xb,
            Xb2,
            Hb,
            Hb2,
            Q,
            KCache,
            VCache,
            AttnScore,
        >,
    >,
    token_embd: Tensor2Imm<'a, T, EMBED, VOCAB, TokenEmbd>,
    output: Tensor2Imm<'a, T, EMBED, VOCAB, Output>,
    output_norm: VectorImm<'a, T, EMBED, OutputNorm>,
    tokenizer: Tokenizer,
}

// LlamaBlock

#[allow(dead_code)]
pub struct LlamaBlock<
    'a,
    TA,
    D,
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
    Xb,
    Xb2,
    Hb,
    Hb2,
    Q,
    KCache,
    VCache,
    AttnScore,
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
    Xb: TensorTypes<T, V<EMBED>>,
    Xb2: TensorTypes<T, V<EMBED>>,
    Hb: TensorTypes<T, V<FF>>,
    Hb2: TensorTypes<T, V<FF>>,
    Q: TensorTypes<T, V<EMBED>>,
    KCache: TensorTypes<T, M<KV, EMBED>>,
    VCache: TensorTypes<T, M<KV, EMBED>>,
    AttnScore: TensorTypes<T, M<CONTEXT, KV>>,
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
    xb: Tensor<'a, true, T, V<EMBED>, Xb>,
    xb2: Tensor<'a, true, T, V<EMBED>, Xb2>,
    hb: Tensor<'a, true, T, V<FF>, Hb>,
    hb2: Tensor<'a, true, T, V<FF>, Hb2>,
    q: Tensor<'a, true, T, V<EMBED>, Q>,
    k_cache: Tensor<'a, true, T, M<KV, EMBED>, KCache>,
    v_cache: Tensor<'a, true, T, M<KV, EMBED>, VCache>,
    attn_score: Tensor<'a, true, T, M<CONTEXT, KV>, AttnScore>,

    phantom0: PhantomData<TA>,
    phantom1: PhantomData<D>,
}

impl<
        'a,
        TA,
        D,
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
        Xb,
        Xb2,
        Hb,
        Hb2,
        Q,
        KCache,
        VCache,
        AttnScore,
    > Instantiable<TA, (&'a D, usize, LlamaParams<T>)>
    for LlamaBlock<
        'a,
        TA,
        D,
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
        Xb,
        Xb2,
        Hb,
        Hb2,
        Q,
        KCache,
        VCache,
        AttnScore,
    >
where
    AttnQ: TensorTypes<T, M<EMBED, EMBED>>,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnQ>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, EMBED>>,

    AttnK: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnK>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, KV>>,

    AttnV: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnV>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, KV>>,

    AttnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, AttnNorm>:
        Instantiable<TA, (&'a D, String)> + TReader<T, V<EMBED>>,

    AttnOutput: TensorTypes<T, M<EMBED, EMBED>>,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnOutput>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, EMBED>>,

    FfnDown: TensorTypes<T, M<FF, EMBED>>,
    Tensor<'a, false, T, M<FF, EMBED>, FfnDown>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<FF, EMBED>>,

    FfnUp: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnUp>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, FF>>,

    FfnGate: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnGate>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, FF>>,

    FfnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, FfnNorm>:
        Instantiable<TA, (&'a D, String)> + TReader<T, V<EMBED>>,

    Xb: TensorTypes<T, V<EMBED>>,
    Tensor<'a, true, T, V<EMBED>, Xb>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<EMBED>>,

    Xb2: TensorTypes<T, V<EMBED>>,
    Tensor<'a, true, T, V<EMBED>, Xb2>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<EMBED>>,

    Hb: TensorTypes<T, V<FF>>,
    Tensor<'a, true, T, V<FF>, Hb>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<FF>>,

    Hb2: TensorTypes<T, V<FF>>,
    Tensor<'a, true, T, V<FF>, Hb2>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<FF>>,

    Q: TensorTypes<T, V<EMBED>>,
    Tensor<'a, true, T, V<EMBED>, Q>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<EMBED>>,

    KCache: TensorTypes<T, M<KV, EMBED>>,
    Tensor<'a, true, T, M<KV, EMBED>, KCache>:
        Instantiable<TA, (&'a D, String)> + TWriter<T, M<KV, EMBED>>,

    VCache: TensorTypes<T, M<KV, EMBED>>,
    Tensor<'a, true, T, M<KV, EMBED>, VCache>:
        Instantiable<TA, (&'a D, String)> + TWriter<T, M<KV, EMBED>>,

    AttnScore: TensorTypes<T, M<CONTEXT, KV>>,
    Tensor<'a, true, T, M<CONTEXT, KV>, AttnScore>:
        Instantiable<TA, (&'a D, String)> + TWriter<T, M<CONTEXT, KV>>,
{
    fn instantiate(
        (model, i, params): (&'a D, usize, LlamaParams<T>),
    ) -> Result<Self, anyhow::Error> {
        // Attention
        let attn_q = Instantiable::instantiate((model, format!("blk.{}.attn_q.weight", i)))?;
        let attn_k = Instantiable::instantiate((model, format!("blk.{}.attn_k.weight", i)))?;
        let attn_v = Instantiable::instantiate((model, format!("blk.{}.attn_v.weight", i)))?;
        let attn_norm = Instantiable::instantiate((model, format!("blk.{}.attn_norm.weight", i)))?;
        let attn_ouput =
            Instantiable::instantiate((model, format!("blk.{}.attn_output.weight", i)))?;

        // Forward network
        let ffn_down = Instantiable::instantiate((model, format!("blk.{}.ffn_down.weight", i)))?;
        let ffn_up = Instantiable::instantiate((model, format!("blk.{}.ffn_up.weight", i)))?;
        let ffn_norm = Instantiable::instantiate((model, format!("blk.{}.ffn_norm.weight", i)))?;
        let ffn_gate = Instantiable::instantiate((model, format!("blk.{}.ffn_gate.weight", i)))?;

        // Mutable
        let xb = Instantiable::instantiate((model, "xb".to_string()))?;
        let xb2 = Instantiable::instantiate((model, "xb2".to_string()))?;
        let hb = Instantiable::instantiate((model, "hb".to_string()))?;
        let hb2 = Instantiable::instantiate((model, "hb2".to_string()))?;
        let q = Instantiable::instantiate((model, "q".to_string()))?;
        let k_cache = Instantiable::instantiate((model, "k_cache".to_string()))?;
        let v_cache = Instantiable::instantiate((model, "v_cache".to_string()))?;
        let attn_score = Instantiable::instantiate((model, "attn_score".to_string()))?;

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
            phantom0: PhantomData,
            phantom1: PhantomData,
        })
    }
}

impl<
        'a,
        TA,
        D: 'a,
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
        Xb,
        Xb2,
        Hb,
        Hb2,
        Q,
        KCache,
        VCache,
        AttnScore,
    >
    LlamaBlock<
        'a,
        TA,
        D,
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
        Xb,
        Xb2,
        Hb,
        Hb2,
        Q,
        KCache,
        VCache,
        AttnScore,
    >
where
    AttnQ: TensorTypes<T, M<EMBED, EMBED>>,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnQ>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, EMBED>>,

    AttnK: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnK>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, KV>>,

    AttnV: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnV>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, KV>>,

    AttnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, AttnNorm>:
        Instantiable<TA, (&'a D, String)> + TReader<T, V<EMBED>>,

    AttnOutput: TensorTypes<T, M<EMBED, EMBED>>,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnOutput>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, EMBED>>,

    FfnDown: TensorTypes<T, M<FF, EMBED>>,
    Tensor<'a, false, T, M<FF, EMBED>, FfnDown>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<FF, EMBED>>,

    FfnUp: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnUp>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, FF>>,

    FfnGate: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnGate>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, FF>>,

    FfnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, FfnNorm>:
        Instantiable<TA, (&'a D, String)> + TReader<T, V<EMBED>>,

    Xb: TensorTypes<T, V<EMBED>>,
    Tensor<'a, true, T, V<EMBED>, Xb>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<EMBED>>,

    Xb2: TensorTypes<T, V<EMBED>>,
    Tensor<'a, true, T, V<EMBED>, Xb2>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<EMBED>>,

    Hb: TensorTypes<T, V<FF>>,
    Tensor<'a, true, T, V<FF>, Hb>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<FF>>,

    Hb2: TensorTypes<T, V<FF>>,
    Tensor<'a, true, T, V<FF>, Hb2>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<FF>>,

    Q: TensorTypes<T, V<EMBED>>,
    Tensor<'a, true, T, V<EMBED>, Q>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<EMBED>>,

    KCache: TensorTypes<T, M<KV, EMBED>>,
    Tensor<'a, true, T, M<KV, EMBED>, KCache>: Instantiable<TA, (&'a D, String)>
        + TWriter<T, M<KV, EMBED>>
        + RowableMut<T, KV, EMBED, KCache>,

    VCache: TensorTypes<T, M<KV, EMBED>>,
    Tensor<'a, true, T, M<KV, EMBED>, VCache>: Instantiable<TA, (&'a D, String)>
        + TWriter<T, M<KV, EMBED>>
        + RowableMut<T, KV, EMBED, VCache>,

    AttnScore: TensorTypes<T, M<CONTEXT, KV>>,
    Tensor<'a, true, T, M<CONTEXT, KV>, AttnScore>: Instantiable<TA, (&'a D, String)>
        + TWriter<T, M<CONTEXT, KV>>
        + RowableMut<T, CONTEXT, KV, AttnScore>,
{
    #[inline(always)]
    pub unsafe fn forward(&mut self, x: &mut VectorMut<T, EMBED>, pos: usize) {
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

        let attn_head_size = self.params.embedding_length / self.params.attention_head_count;
        let kv_mul = 4; // TODO: remove hardcoded
        debug_assert!(attn_head_size == 128);

        // q, v and k for this position
        let q = &mut self.q;
        {
            let k = &mut self.k_cache.row(pos);
            let v = &mut self.v_cache.row(pos);
            matmul(q, &mut self.attn_q, xb);
            matmul(k, &mut self.attn_k, xb);
            matmul(v, &mut self.attn_v, xb);

            //let mut qw = q.writer();
            let mut qw = q.writer();
            let mut kw = k.writer();

            // RoPE
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
        }

        // Multihead attention
        for h in 0..self.params.attention_head_count {
            let attn_score = &mut self.attn_score.row(h);
            {
                let mut attn_score_w = attn_score.writer();
                let q_offset = h * attn_head_size;
                for t in 0..(pos + 1) {
                    let mut score = T::zero();
                    let mut qw = q.writer();
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
                        xbw.set(
                            xb_offset + i,
                            xbw.get(xb_offset + i) + a * v_w.get(v_offset + i),
                        );
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
        TA,
        D,
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
        const CONTEXT: usize,
        Xb,
        Xb2,
        Hb,
        Hb2,
        Q,
        KCache,
        VCache,
        AttnScore,
    > Instantiable<TA, (&'a D, &str)>
    for Llama<
        'a,
        TA,
        D,
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
        CONTEXT,
        Xb,
        Xb2,
        Hb,
        Hb2,
        Q,
        KCache,
        VCache,
        AttnScore,
    >
where
    AttnQ: TensorTypes<T, M<EMBED, EMBED>>,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnQ>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, EMBED>>,

    AttnK: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnK>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, KV>>,

    AttnV: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnV>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, KV>>,

    AttnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, AttnNorm>:
        Instantiable<TA, (&'a D, String)> + TReader<T, V<EMBED>>,

    AttnOutput: TensorTypes<T, M<EMBED, EMBED>>,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnOutput>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, EMBED>>,

    FfnDown: TensorTypes<T, M<FF, EMBED>>,
    Tensor<'a, false, T, M<FF, EMBED>, FfnDown>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<FF, EMBED>>,

    FfnUp: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnUp>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, FF>>,

    FfnGate: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnGate>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, FF>>,

    FfnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, FfnNorm>:
        Instantiable<TA, (&'a D, String)> + TReader<T, V<EMBED>>,

    Xb: TensorTypes<T, V<EMBED>>,
    Tensor<'a, true, T, V<EMBED>, Xb>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<EMBED>>,

    Xb2: TensorTypes<T, V<EMBED>>,
    Tensor<'a, true, T, V<EMBED>, Xb2>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<EMBED>>,

    Hb: TensorTypes<T, V<FF>>,
    Tensor<'a, true, T, V<FF>, Hb>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<FF>>,

    Hb2: TensorTypes<T, V<FF>>,
    Tensor<'a, true, T, V<FF>, Hb2>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<FF>>,

    Q: TensorTypes<T, V<EMBED>>,
    Tensor<'a, true, T, V<EMBED>, Q>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<EMBED>>,

    KCache: TensorTypes<T, M<KV, EMBED>>,
    Tensor<'a, true, T, M<KV, EMBED>, KCache>: Instantiable<TA, (&'a D, String)>
        + TWriter<T, M<KV, EMBED>>
        + RowableMut<T, KV, EMBED, KCache>,

    VCache: TensorTypes<T, M<KV, EMBED>>,
    Tensor<'a, true, T, M<KV, EMBED>, VCache>: Instantiable<TA, (&'a D, String)>
        + TWriter<T, M<KV, EMBED>>
        + RowableMut<T, KV, EMBED, VCache>,

    AttnScore: TensorTypes<T, M<CONTEXT, KV>>,
    Tensor<'a, true, T, M<CONTEXT, KV>, AttnScore>: Instantiable<TA, (&'a D, String)>
        + TWriter<T, M<CONTEXT, KV>>
        + RowableMut<T, CONTEXT, KV, AttnScore>,

    TokenEmbd: TensorTypes<T, M<EMBED, VOCAB>>,
    Tensor<'a, false, T, M<EMBED, VOCAB>, TokenEmbd>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, VOCAB>>,

    Output: TensorTypes<T, M<EMBED, VOCAB>>,
    Tensor<'a, false, T, M<EMBED, VOCAB>, Output>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, VOCAB>>,

    OutputNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, OutputNorm>:
        Instantiable<TA, (&'a D, String)> + TReader<T, V<EMBED>>,

    LlamaParams<T>: Instantiable<TA, &'a D>,
{
    fn instantiate((model, tokenizer_path): (&'a D, &str)) -> Result<Self, anyhow::Error> {
        //let header = &model.header;

        // Huggingface tokenizer
        // TODO: implement a local algorithm using using data in GGUF
        let tokenizer = match Tokenizer::from_file(tokenizer_path) {
            Ok(tokenizer) => Ok(tokenizer),
            Err(err) => Err(anyhow!("could not create tokenizer: {}", err.to_string())),
        }?;

        let params: LlamaParams<T> = Instantiable::instantiate(model)?;

        if params.vocab_size != VOCAB {
            return Err(anyhow!("'llama.vocab_size' doesn't match the static value"));
        }

        let token_embd = Instantiable::instantiate((model, "token_embd.weight".to_string()))?;
        let output = Instantiable::instantiate((model, "output.weight".to_string()))?;
        let output_norm = Instantiable::instantiate((model, "output_norm.weight".to_string()))?;

        let blocks: Result<Vec<_>, anyhow::Error> = (0..params.block_count)
            .into_iter()
            .map(|i| LlamaBlock::instantiate((model, i, params)))
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
        TA,
        D: 'a,
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
        const CONTEXT: usize,
        Xb,
        Xb2,
        Hb,
        Hb2,
        Q,
        KCache,
        VCache,
        AttnScore,
    > LLM<'a, TA, T, u32, ModelFile, EMBED, VOCAB>
    for Llama<
        'a,
        TA,
        D,
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
        CONTEXT,
        Xb,
        Xb2,
        Hb,
        Hb2,
        Q,
        KCache,
        VCache,
        AttnScore,
    >
where
    AttnQ: TensorTypes<T, M<EMBED, EMBED>>,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnQ>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, EMBED>>,

    AttnK: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnK>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, KV>>,

    AttnV: TensorTypes<T, M<EMBED, KV>>,
    Tensor<'a, false, T, M<EMBED, KV>, AttnV>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, KV>>,

    AttnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, AttnNorm>:
        Instantiable<TA, (&'a D, String)> + TReader<T, V<EMBED>>,

    AttnOutput: TensorTypes<T, M<EMBED, EMBED>>,
    Tensor<'a, false, T, M<EMBED, EMBED>, AttnOutput>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, EMBED>>,

    FfnDown: TensorTypes<T, M<FF, EMBED>>,
    Tensor<'a, false, T, M<FF, EMBED>, FfnDown>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<FF, EMBED>>,

    FfnUp: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnUp>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, FF>>,

    FfnGate: TensorTypes<T, M<EMBED, FF>>,
    Tensor<'a, false, T, M<EMBED, FF>, FfnGate>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, FF>>,

    FfnNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, FfnNorm>:
        Instantiable<TA, (&'a D, String)> + TReader<T, V<EMBED>>,

    Xb: TensorTypes<T, V<EMBED>>,
    Tensor<'a, true, T, V<EMBED>, Xb>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<EMBED>>,

    Xb2: TensorTypes<T, V<EMBED>>,
    Tensor<'a, true, T, V<EMBED>, Xb2>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<EMBED>>,

    Hb: TensorTypes<T, V<FF>>,
    Tensor<'a, true, T, V<FF>, Hb>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<FF>>,

    Hb2: TensorTypes<T, V<FF>>,
    Tensor<'a, true, T, V<FF>, Hb2>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<FF>>,

    Q: TensorTypes<T, V<EMBED>>,
    Tensor<'a, true, T, V<EMBED>, Q>: Instantiable<TA, (&'a D, String)> + TWriter<T, V<EMBED>>,

    KCache: TensorTypes<T, M<KV, EMBED>>,
    Tensor<'a, true, T, M<KV, EMBED>, KCache>: Instantiable<TA, (&'a D, String)>
        + TWriter<T, M<KV, EMBED>>
        + RowableMut<T, KV, EMBED, KCache>,

    VCache: TensorTypes<T, M<KV, EMBED>>,
    Tensor<'a, true, T, M<KV, EMBED>, VCache>: Instantiable<TA, (&'a D, String)>
        + TWriter<T, M<KV, EMBED>>
        + RowableMut<T, KV, EMBED, VCache>,

    AttnScore: TensorTypes<T, M<CONTEXT, KV>>,
    Tensor<'a, true, T, M<CONTEXT, KV>, AttnScore>: Instantiable<TA, (&'a D, String)>
        + TWriter<T, M<CONTEXT, KV>>
        + RowableMut<T, CONTEXT, KV, AttnScore>,

    TokenEmbd: TensorTypes<T, M<EMBED, VOCAB>>,
    Tensor<'a, false, T, M<EMBED, VOCAB>, TokenEmbd>: Instantiable<TA, (&'a D, String)>
        + TReader<T, M<EMBED, VOCAB>>
        + Rowable<T, EMBED, VOCAB, TokenEmbd>,

    Output: TensorTypes<T, M<EMBED, VOCAB>>,
    Tensor<'a, false, T, M<EMBED, VOCAB>, Output>:
        Instantiable<TA, (&'a D, String)> + TReader<T, M<EMBED, VOCAB>>,

    OutputNorm: TensorTypes<T, V<EMBED>>,
    Tensor<'a, false, T, V<EMBED>, OutputNorm>:
        Instantiable<TA, (&'a D, String)> + TReader<T, V<EMBED>>,
{
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
        let mut x2: VectorMut<T, EMBED> = VectorMut::new_vector();
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
