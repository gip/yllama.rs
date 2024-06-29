pub mod tensor;
pub mod function {
    pub use super::*;
}
use rand::distributions::Slice;
use tensor::*;

use half::f16;
use memmap2::Mmap;
use num_traits::float::Float;
use std::ops::{Index, IndexMut};
use std::rc::Rc;

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

pub unsafe fn matmul<'a, T, M0, V0, V1, const D0: usize, const D1: usize>(
    v1: &'a mut V1,
    m0: &'a M0,
    v0: &'a V0,
) where
    T: Float + 'a,
    M0: TReader<T, MATRIX<D0, D1>>,
    V0: TReader<T, VECTOR<D0>>,
    V1: TWriter<T, VECTOR<D1>>,
    M0::Reader<'a>: Index<(usize, usize), Output = T>,
    V0::Reader<'a>: Index<usize, Output = T>,
    V1::Writer<'a>: IndexMut<usize, Output = T> + Sized,
{
    let m0 = m0.reader();
    let v0 = v0.reader();
    let mut v1 = v1.writer();
    for i in 0..D1 {
        let mut r = T::zero();
        for j in 0..D0 {
            let m_val: T = m0[(i, j)];
            let v_val: T = v0[j];
            r = r + m_val * v_val;
        }
        v1[i] = r;
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
    let ys = y.reader();
    let mut xs = x.writer();
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
