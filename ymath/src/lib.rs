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
