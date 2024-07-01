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

pub unsafe fn matmul<T: Float, const D0: usize, const D1: usize>(
    v1: &mut impl TWriter<T, VECTOR<D1>>,
    m0: &impl TReader<T, MATRIX<D0, D1>>,
    v0: &impl TReader<T, VECTOR<D0>>,
) {
    let m0 = m0.reader();
    let v0 = v0.reader();
    let mut v1 = v1.writer();
    for i in 0..D1 {
        let mut r = T::zero();
        for j in 0..D0 {
            r = r + m0.get((i, j)) * v0.get(j);
        }
        v1.set(i, r);
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

pub fn max<'a, T: 'a, const D0: usize>(x: &'a mut impl TReader<T, VECTOR<D0>>) -> (usize, T)
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
