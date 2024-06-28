pub mod tensor;
pub mod function {
    pub use super::*;
}
use tensor::*;

use half::f16;
use memmap2::Mmap;
use num_traits::float::Float;
use std::ops::{Index, IndexMut};
use std::rc::Rc;

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

#[derive(Clone, Copy)]
pub struct BlockQ4K {
    d: f16,
    dmin: f16,
    scales: [u8; K_SCALE_SIZE],
    qs: [u8; QK_K / 2],
}

#[derive(Clone, Copy)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    scales: [i8; QK_K / 16],
    d: f16,
}

#[inline(always)]
fn set<T>(slice: &mut [T], index: usize, value: T) {
    if cfg!(debug_assertions) {
        slice[index] = value;
    } else {
        unsafe {
            *slice.get_unchecked_mut(index) = value;
        }
    }
}

#[inline(always)]
fn get_scale_min_k4(j: usize, q: &[u8; 12], d: &mut u8, m: &mut u8) {
    if j < 4 {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Q4_K
// Block is 144 bytes and quantizes 256 f32 weights -> 4.5 bits per weight
pub fn dequantize_row_q4_k(x: &[BlockQ4K], y: &mut Vec<f32>, k: usize) -> usize {
    assert!(k % QK_K == 0);
    let nb: usize = k / QK_K;
    let mut ycount: usize = 0;

    for i in 0..nb {
        let q = x[i].qs;
        let mut iq: usize = 0;

        let d: f32 = x[i].d.to_f32();
        let min: f32 = x[i].dmin.to_f32();

        let mut is = 0;
        let mut sc: u8 = 0;
        let mut m: u8 = 0;
        for _ in (0..QK_K).step_by(64) {
            get_scale_min_k4(is + 0, &x[i].scales, &mut sc, &mut m);
            let d1: f32 = d * sc as f32;
            let m1: f32 = min * m as f32;
            get_scale_min_k4(is + 1, &x[i].scales, &mut sc, &mut m);
            let d2: f32 = d * sc as f32;
            let m2: f32 = min * m as f32;
            for l in 0..32 {
                let val: f32 = d1 * (q[iq + l] & 0xF) as f32 - m1;
                set(y, ycount, val);
                ycount = ycount + 1;
            }
            for l in 0..32 {
                let val: f32 = d2 * (q[iq + l] >> 4) as f32 - m2;
                set(y, ycount, val);
                ycount = ycount + 1;
            }
            iq += 32;
            is += 2;
        }
    }
    ycount
}

// Q6_K
// Block is 210 bytes and quantizes 256 f32 weights -> ~6.5 bits per weight
// https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
pub fn dequantize_row_q6_k(x: &[BlockQ6K], y: &mut Vec<f32>, k: usize) -> usize {
    assert!(k % QK_K == 0);
    let nb: usize = k / QK_K;
    let mut ycount: usize = 0;

    for i in 0..nb {
        let d = x[i].d.to_f32();
        let mut ql = 0;
        let mut qh = 0;
        let mut sc = 0;
        for _ in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let q1: i8 = ((x[i].ql[ql + l + 0] & 0xF) as i8
                    | (((x[i].qh[qh + l] >> 0) & 3) << 4) as i8)
                    - 32;
                let q2: i8 = ((x[i].ql[ql + l + 32] & 0xF) as i8
                    | (((x[i].qh[qh + l] >> 2) & 3) << 4) as i8)
                    - 32; // TODO: check
                let q3: i8 = ((x[i].ql[ql + l + 0] >> 4) as i8
                    | (((x[i].qh[qh + l] >> 4) & 3) << 4) as i8)
                    - 32;
                let q4: i8 = ((x[i].ql[ql + l + 32] >> 4) as i8
                    | (((x[i].qh[qh + l] >> 6) & 3) << 4) as i8)
                    - 32;
                set(
                    y,
                    ycount + l + 0,
                    d * x[i].scales[sc + is + 0] as f32 * q1 as f32,
                );
                set(
                    y,
                    ycount + l + 32,
                    d * x[i].scales[sc + is + 2] as f32 * q2 as f32,
                );
                set(
                    y,
                    ycount + l + 64,
                    d * x[i].scales[sc + is + 4] as f32 * q3 as f32,
                );
                set(
                    y,
                    ycount + l + 96,
                    d * x[i].scales[sc + is + 6] as f32 * q4 as f32,
                );
            }
            ycount += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    ycount
}

pub fn softmax<T: Float, const D0: usize>(v: &mut impl TWriter<T, VECTOR<D0>>, size: usize) {
    let mut writer = v.writer();
    let v = writer.writing();
    debug_assert!(size < D0);
    let mut max = v[0];
    for i in 1..size {
        if v[i] > max {
            max = v[i]
        }
    }
    let mut sum = T::zero();
    for i in 0..size {
        let val = T::exp(v[i] - max);
        v[i] = val;
        sum = sum + val;
    }
    for i in 0..size {
        v[i] = v[i] / sum;
    }
}

pub unsafe fn matmul<T: Float, const D0: usize, const D1: usize>(
    v1: &mut impl TWriter<T, VECTOR<D1>>,
    m0: &impl TReader<T, MATRIX<D0, D1>>,
    v0: &impl TReader<T, VECTOR<D0>>,
) {
    let m0_reader = m0.reader();
    let m0_slice = m0_reader.reading();
    let v0_reader = v0.reader();
    let v0_slice = v0_reader.reading();
    let mut v1_writer = v1.writer();
    let v1_slice = v1_writer.writing();
    for i in 0..D1 {
        let mut r = T::zero();
        for j in 0..D0 {
            r = r + m0_slice[i * D0 + j] * v0_slice[j];
        }
        v1_slice[i] = r;
    }
}

pub fn rmsnorm<'a, 'b, T: Float, const D0: usize>(
    xout: &'a mut impl TWriter<T, VECTOR<D0>>,
    xin: &'a impl TReader<T, VECTOR<D0>>,
    w: &'a impl TReader<T, VECTOR<D0>>,
    epsilon: T,
) {
    let xin = xin.reader();
    let xins = xin.reading();
    let w = w.reader();
    let ws = w.reading();
    let mut xout = xout.writer();
    let xouts = xout.writing();
    let mut ss = T::zero();
    for i in 0..D0 {
        ss = ss + xins[i] * xins[i];
    }
    ss = ss / T::from(D0).unwrap() + epsilon;
    ss = T::one() / ss.sqrt();
    for i in 0..D0 {
        xouts[i] = ws[i] * (ss * xins[i]);
    }
}

//
pub fn acc<'a, T: Float, const D0: usize>(
    x: &'a mut impl TWriter<T, VECTOR<D0>>,
    y: &'a impl TReader<T, VECTOR<D0>>,
) {
    let mut x = x.writer();
    let xs = x.writing();
    let y = y.reader();
    let ys = y.reading();
    for i in 0..D0 {
        xs[i] = xs[i] + ys[i];
    }
}

pub fn cp<'a, T: Float, const D0: usize>(
    x: &'a mut impl TWriter<T, VECTOR<D0>>,
    y: &'a impl TReader<T, VECTOR<D0>>,
) {
    let y = y.reader();
    let ys = y.reading();
    let mut x = x.writer();
    let xs = x.writing();
    for i in 0..D0 {
        xs[i] = ys[i];
    }
}

pub fn max<'a, T: 'a, const D0: usize>(x: &'a mut impl TReader<T, VECTOR<D0>>) -> (usize, T)
where
    T: Float,
{
    let x = x.reader();
    let slice = x.reading();
    let mut i = 0;
    let mut m = slice[0];
    for j in 1..D0 {
        if slice[j] > m {
            i = j;
            m = slice[j];
        }
    }
    (i, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    pub fn misc() {
        let block = BlockQ4K {
            d: f16::from_f32(1.0012),
            dmin: f16::from_f32(-1.12),
            scales: [10; 12],
            qs: [255; 128],
        };

        let block6 = BlockQ6K {
            ql: [129; QK_K / 2],
            qh: [200; QK_K / 4],
            scales: [-1; QK_K / 16],
            d: f16::from_f32(100.55),
        };

        let mut res = vec![0.0; 256];

        let blocks = [block; 1];
        dequantize_row_q4_k(&blocks, &mut res, 256);
        let q4_bytes = mem::size_of::<[BlockQ4K; 1]>();
        assert!(q4_bytes == 144);
        println!("res {:?}", res);

        let blocks6 = [block6; 1];
        dequantize_row_q6_k(&blocks6, &mut res, 256);
        println!("res {:?}", res);

        let q6_bytes = mem::size_of::<[BlockQ6K; 1]>();
        assert!(q6_bytes == 210);

        println!("q4_bytes = {:?}", q4_bytes);
        println!("q6_bytes = {:?}", q6_bytes);
    }
}
