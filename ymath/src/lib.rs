#![feature(specialization)]
#![feature(portable_simd)]

pub mod tensor;
pub mod function {
    pub use super::*;
}
use tensor::*;

use half::f16;
use memmap2::Mmap;
use num_traits::float::Float;
use std::rc::Rc;

pub fn softmax<T: Float, const D0: usize>(v: &mut impl TWriter<T, VECTOR<D0>>, size: usize) {
    let mut v = v.writer();
    debug_assert!(size < D0);
    let mut max = v.get(0);
    for i in 1..size {
        if v.get(i) > max {
            max = v.get(i)
        }
    }
    let mut sum = T::zero();
    for i in 0..size {
        let val = T::exp(v.get(i) - max);
        v.set(i, val);
        sum = sum + val;
    }
    for i in 0..size {
        v.set(i, v.get(i) / sum);
    }
}

pub trait Matmul
where
    Self: Float,
{
    unsafe fn matmul<const D0: usize, const D1: usize>(
        v1: &mut impl TWriter<Self, VECTOR<D1>>,
        m0: &impl TReader<Self, MATRIX<D0, D1>>,
        v0: &impl TReader<Self, VECTOR<D0>>,
    );
}

impl<T: Float> Matmul for T {
    default unsafe fn matmul<const D0: usize, const D1: usize>(
        v1: &mut impl TWriter<Self, VECTOR<D1>>,
        m0: &impl TReader<Self, MATRIX<D0, D1>>,
        v0: &impl TReader<Self, VECTOR<D0>>,
    ) {
        let m0 = m0.reader();
        let v0 = v0.reader();
        let mut v1 = v1.writer();
        for i in 0..D1 {
            let mut r = Self::zero();
            for j in 0..D0 {
                r = r + m0.get((i, j)) * v0.get(j);
            }
            v1.set(i, r);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Matmul for f32 {
    unsafe fn matmul<const D0: usize, const D1: usize>(
        v1: &mut impl TWriter<f32, VECTOR<D1>>,
        m0: &impl TReader<f32, MATRIX<D0, D1>>,
        v0: &impl TReader<f32, VECTOR<D0>>,
    ) {
        use std::simd::num::SimdFloat;
        use std::simd::*;

        const SIMD_WIDTH: usize = 8; // Up to 8 lanes
        type SimdVec = Simd<f32, SIMD_WIDTH>;

        let m0 = m0.reader();
        let v0 = v0.reader();
        let mut v1 = v1.writer();

        for i in 0..D1 {
            let mut r: SimdVec = Simd::splat(0.0);
            let mut j = 0;

            while j + SIMD_WIDTH <= D0 {
                let a: SimdVec = Simd::from_slice(&[
                    m0.get((i, j)),
                    m0.get((i, j + 1)),
                    m0.get((i, j + 2)),
                    m0.get((i, j + 3)),
                    m0.get((i, j + 4)),
                    m0.get((i, j + 5)),
                    m0.get((i, j + 6)),
                    m0.get((i, j + 7)),
                ]);
                let b: SimdVec = Simd::from_slice(&[
                    v0.get(j),
                    v0.get(j + 1),
                    v0.get(j + 2),
                    v0.get(j + 3),
                    v0.get(j + 4),
                    v0.get(j + 5),
                    v0.get(j + 6),
                    v0.get(j + 7),
                ]);
                r += a * b;
                j += SIMD_WIDTH;
            }

            let mut sum = r.reduce_sum();
            while j < D0 {
                sum += m0.get((i, j)) * v0.get(j);
                j += 1;
            }
            v1.set(i, sum);
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl Matmul for f32 {
    #[target_feature(enable = "simd128")]
    unsafe fn matmul<const D0: usize, const D1: usize>(
        v1: &mut impl TWriter<f32, VECTOR<D1>>,
        m0: &impl TReader<f32, MATRIX<D0, D1>>,
        v0: &impl TReader<f32, VECTOR<D0>>,
    ) {
        use std::arch::wasm32::*;

        let m0 = m0.reader();
        let v0 = v0.reader();
        let mut v1 = v1.writer();
        for i in 0..D1 {
            let mut r = f32x4(0.0, 0.0, 0.0, 0.0);
            for j in (0..D0).step_by(4) {
                let a = f32x4(
                    m0.get((i, j + 0)),
                    m0.get((i, j + 1)),
                    m0.get((i, j + 2)),
                    m0.get((i, j + 3)),
                );
                let b = f32x4(v0.get(j + 0), v0.get(j + 1), v0.get(j + 2), v0.get(j + 3));
                r = f32x4_add(r, f32x4_mul(a, b));
            }
            let sum = f32x4_extract_lane::<0>(r)
                + f32x4_extract_lane::<1>(r)
                + f32x4_extract_lane::<2>(r)
                + f32x4_extract_lane::<3>(r);
            v1.set(i, sum);
        }
    }
}

pub fn rmsnorm<'a, 'b, T: Float, const D0: usize>(
    xout: &'a mut impl TWriter<T, VECTOR<D0>>,
    xin: &'a impl TReader<T, VECTOR<D0>>,
    w: &'a impl TReader<T, VECTOR<D0>>,
    epsilon: T,
) {
    let xin = xin.reader();
    let w = w.reader();
    let mut xout = xout.writer();
    let mut ss = T::zero();
    for i in 0..D0 {
        ss = ss + xin.get(i) * xin.get(i);
    }
    ss = ss / T::from(D0).unwrap() + epsilon;
    ss = T::one() / ss.sqrt();
    for i in 0..D0 {
        xout.set(i, w.get(i) * (ss * xin.get(i)));
    }
}

pub fn acc<'a, T: Float, const D0: usize>(
    x: &'a mut impl TWriter<T, VECTOR<D0>>,
    y: &'a impl TReader<T, VECTOR<D0>>,
) {
    let mut x = x.writer();
    let y = y.reader();
    for i in 0..D0 {
        x.set(i, x.get(i) + y.get(i));
    }
}

pub fn cp<'a, T: Float, const D0: usize>(
    x: &'a mut impl TWriter<T, VECTOR<D0>>,
    y: &'a impl TReader<T, VECTOR<D0>>,
) {
    let y = y.reader();
    let mut x = x.writer();
    for i in 0..D0 {
        x.set(i, y.get(i))
    }
}

pub fn max<'a, T: 'a, const D0: usize>(x: &'a impl TReader<T, VECTOR<D0>>) -> (usize, T)
where
    T: Float,
{
    let x = x.reader();
    let mut i = 0;
    let mut m = x.get(0);
    for j in 1..D0 {
        if x.get(j) > m {
            i = j;
            m = x.get(j);
        }
    }
    (i, m)
}
