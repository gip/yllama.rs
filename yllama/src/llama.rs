use crate::model::LLM;
use anyhow::anyhow;
use yloader::*;
use ymath::*;
use tokenizers::tokenizer::Tokenizer;

type ModelDescription<'a> = ModelFile<GGUFFile<MemLayout<'a, f32>>>;

#[derive(Debug, Clone, Copy)]
struct LlamaParams<T = usize, U = f32> {
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
    max_seq_len: T,                      // This could be context_length or lower to save space
    attention_kv_length: T,
}

#[derive(Debug)]
#[allow(dead_code)]
struct LlamaBlock<'a, T = f32> {
    i: usize,
    params: LlamaParams,

    // Model weights
    attn_q: Tensor2<'a, T>,
    attn_k: Tensor2<'a, T>,
    attn_v: Tensor2<'a, T>,
    attn_norm: Vector<'a, T>,
    attn_ouput: Tensor2<'a, T>,
    ffn_down: Tensor2<'a, T>,
    ffn_up: Tensor2<'a, T>,
    ffn_norm: Vector<'a, T>,
    ffn_gate: Tensor2<'a, T>,

    // Block state
    xb: VectorMut<'a, T>,
    xb2: VectorMut<'a, T>,
    hb: VectorMut<'a, T>,
    hb2: VectorMut<'a, f32>,
    q: VectorMut<'a, T>,
    k_cache: Tensor2Mut<'a, T>,
    v_cache: Tensor2Mut<'a, T>,
    attn_score: Tensor2Mut<'a, T>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Llama<'a, T = f32> {
    params: LlamaParams,
    blocks: Vec<LlamaBlock<'a>>,
    token_embd: Tensor2<'a, T>,
    output: Tensor2<'a, T>,
    output_norm: Vector<'a, T>,
    // tokens: Vec<&'a String>,
    tokenizer: Tokenizer,
}

macro_rules! trans {
    ($func:ident($($arg:expr),*)) => {
        $func($($arg),*);
    };
}

impl<'a> LlamaBlock<'a, f32> {
    fn new(
        model: &'a ModelDescription,
        i: usize,
        params: LlamaParams,
    ) -> Result<LlamaBlock<'a, f32>, anyhow::Error> {
        macro_rules! build_tensor {
            ($s:expr) => {{
                let name = format!($s, i);
                model
                    .model
                    .tensors
                    .get(&name)
                    .ok_or_else(|| anyhow!("tensor {} not found", name))
                    .map(|x| x.to_tensor2())?
            }};
        }

        macro_rules! build_vector {
            ($s:expr) => {{
                let name = format!($s, i);
                model
                    .model
                    .tensors
                    .get(&name)
                    .ok_or_else(|| anyhow!("vector {} not found", name))
                    .map(|x| x.to_vector())?
            }};
        }

        // Attention
        let attn_q = build_tensor!("blk.{}.attn_q.weight");
        let attn_k = build_tensor!("blk.{}.attn_k.weight");
        let attn_v = build_tensor!("blk.{}.attn_v.weight");
        let attn_norm = build_vector!("blk.{}.attn_norm.weight");
        let attn_ouput = build_tensor!("blk.{}.attn_output.weight");

        // Forward network
        let ffn_down = build_tensor!("blk.{}.ffn_down.weight");
        let ffn_up = build_tensor!("blk.{}.ffn_up.weight");
        let ffn_norm = build_vector!("blk.{}.ffn_norm.weight");
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
    unsafe fn forward(&mut self, x: &mut VectorMut<f32>, pos: usize) {
        let xb = &mut self.xb;
        let xb2 = &mut self.xb2;
        let hb = &mut self.hb;
        let hb2 = &mut self.hb2;

        trans!(rmsnorm(
            xb,
            x,
            &self.attn_norm,
            self.params.attention_layer_norm_rms_epsilon
        ));

        // q, v and k for this position
        let q = &mut self.q;
        let k = &mut self.k_cache.row(pos);
        let v = &mut self.v_cache.row(pos);
        matmul(q, &self.attn_q, xb);
        matmul(k, &self.attn_k, xb);
        matmul(v, &self.attn_v, xb);

        // RoPE
        let attn_head_size = self.params.embedding_length / self.params.attention_head_count;
        let kv_mul = 4; // TODO: remove hardcoded
        debug_assert!(attn_head_size == 128);
        for i in 0..self.params.attention_head_count {
            for j in (0..attn_head_size).step_by(2) {
                let freq =
                    1.0 / f32::powf(self.params.rope_freq_base, j as f32 / attn_head_size as f32);
                let val = pos as f32 * freq;
                let fcr = f32::cos(val);
                let fci = f32::sin(val);
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
                let mut score = 0.0;
                let k = &mut self.k_cache.row(t);
                let k_offset = (h / kv_mul) * attn_head_size;
                for i in 0..attn_head_size {
                    score += q[q_offset + i] * k[k_offset + i];
                }
                score /= f32::sqrt(attn_head_size as f32);
                attn_score[t] = score;
            }

            // Softmax from 0..p inclusively
            softmax(attn_score, pos + 1);

            // Weighted sum of the values, store back into xb
            let xb_offset = h * attn_head_size;
            for i in 0..attn_head_size {
                xb[xb_offset + i] = 0.0; // TODO: Optimize? Can we memset?
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
        matmul(xb2, &self.attn_ouput, xb);

        // Residual
        acc(x, xb2);

        // Ffn rmsnorm
        trans!(rmsnorm(
            xb,
            x,
            &self.ffn_norm,
            self.params.attention_layer_norm_rms_epsilon
        ));

        // Non-linearity
        matmul(hb, &self.ffn_gate, xb);
        matmul(hb2, &self.ffn_up, xb);
        for i in 0..self.params.feed_forward_length {
            let mut val = hb[i];
            val *= 1.0 / (1.0 + f32::exp(-val));
            val *= hb2[i];
            hb[i] = val;
        }

        // Ffn output
        matmul(xb, &self.ffn_down, hb);

        // Residual
        acc(x, xb);
    }
}

fn conv_err(b: Box<dyn std::error::Error + Send + Sync>) -> Box<dyn std::error::Error> {
    b
}

impl<'a> Llama<'a> {
    fn new(model: &'a ModelDescription, tokenizer_path: &str) -> Result<Llama<'a>, anyhow::Error> {
        let header = &model.model.header;

        // Huggingface tokenizer
        // TODO: implement a local algorithm using using data in GGUF
        let tokenizer = match Tokenizer::from_file(tokenizer_path) {
            Ok(tokenizer) => Ok(tokenizer),
            Err(err) => Err(anyhow!("could not create tokenizer: {}", err.to_string())),
        }?;

        let embedding_length = header_find_usize(header, "llama.embedding_length")?;
        let attention_head_count_kv = header_find_usize(header, "llama.attention.head_count_kv")?;
        let attention_head_count = header_find_usize(header, "llama.attention.head_count")?;
        let params = LlamaParams {
            block_count: header_find_usize(header, "llama.block_count")?,
            _context_length: header_find_usize(header, "llama.context_length")?,
            embedding_length,
            feed_forward_length: header_find_usize(header, "llama.feed_forward_length")?,
            attention_head_count,
            attention_head_count_kv,
            attention_layer_norm_rms_epsilon: header_find_f32(
                header,
                "llama.attention.layer_norm_rms_epsilon",
            )?,
            rope_freq_base: header_find_f32(header, "llama.rope.freq_base")?,
            _rope_dimension_count: header_find_usize(header, "llama.rope.dimension_count")?,
            vocab_size: header_find_usize(header, "llama.vocab_size")?,
            max_seq_len: 2048, // Where does it come from?
            attention_kv_length: embedding_length * attention_head_count_kv / attention_head_count,
        };

        macro_rules! build_tensor {
            ($s:expr) => {{
                let name = $s;
                model
                    .model
                    .tensors
                    .get(name)
                    .ok_or_else(|| anyhow!("tensor {} not found", name))
                    .map(|x| x.to_tensor2())?
            }};
        }

        macro_rules! build_vector {
            ($s:expr) => {{
                let name = $s;
                model
                    .model
                    .tensors
                    .get(name)
                    .ok_or_else(|| anyhow!("vector {} not found", name))
                    .map(|x| x.to_vector())?
            }};
        }

        let token_embd = build_tensor!("token_embd.weight");
        let output = build_tensor!("output.weight");
        let output_norm = build_vector!("output_norm.weight");

        let blocks: Result<Vec<LlamaBlock>, anyhow::Error> = (0..params.block_count)
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

impl<'a> LLM<'a, f32, u32, ModelDescription<'a>> for Llama<'a> {
    fn build<'b>(model: &'a ModelDescription, tokenizer_path: &str) -> Result<Self, anyhow::Error> {
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

    fn embed(&self, x: &mut VectorMut<f32>, token: u32, _pos: usize) {
        cp(x, &self.token_embd.row(token as usize))
    }

    fn decode(&self, tokens: &Vec<u32>) -> String {
        self.tokenizer.decode(tokens, false).unwrap()
    }

    unsafe fn forward(&mut self, x: &mut VectorMut<f32>, pos: usize) {
        self.blocks
            .iter_mut()
            .for_each(|block| block.forward(x, pos));
    }

    unsafe fn block_forward(&mut self, x: &mut VectorMut<f32>, pos: usize, block: usize) {
        self.blocks[block].forward(x, pos)
    }

    unsafe fn logits(&self, logits: &mut VectorMut<f32>, x: &VectorMut<f32>) {
        // Final rmsnorm
        let mut x2 = VectorMut::new(self.params.embedding_length);
        rmsnorm(
            &mut x2,
            x,
            &self.output_norm,
            self.params.attention_layer_norm_rms_epsilon,
        );
        // Last act: logits
        matmul(logits, &self.output, &x2);
    }
}
