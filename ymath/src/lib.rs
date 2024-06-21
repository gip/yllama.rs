use core::ops::Add;
use core::ops::Mul;
use half::f16;
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::ops::{Index, IndexMut};

// Memory layout
#[derive(Clone)]
pub enum MemLayout<'a, T> {
    MmapLayout { slice: &'a [T] },
    VecLayout { vec: Vec<T> },
}

impl<'a, T> MemLayout<'a, T> {
    pub fn to_slice(&self) -> &[T] {
        match self {
            MemLayout::MmapLayout { slice } => slice,
            MemLayout::VecLayout { vec } => vec.as_slice(),
        }
    }
}

impl<'a, T> Debug for MemLayout<'a, T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        let s = match self {
            MemLayout::MmapLayout { .. } => "<MmapLayout>",
            MemLayout::VecLayout { .. } => "<VecLayout>",
        };
        let _ = formatter.pad(s);
        Ok(())
    }
}

// Tensor
pub trait D<const DIM: usize> {
    fn shape(&self) -> [usize; DIM];
}

#[derive(Debug)]
pub struct Tensor<'a, T, const DIM: usize> {
    pub shape: [usize; DIM],
    pub vec: Option<Vec<T>>,
    pub slice: &'a [T],
}

impl<'a, T, const DIM: usize> D<DIM> for Tensor<'a, T, DIM> {
    fn shape(&self) -> [usize; DIM] {
        self.shape
    }
}

impl<'a, T, const DIM: usize> Clone for  Tensor<'a, T, DIM> 
where T : Clone {
    fn clone(&self) -> Self {
        let mut vec = Vec::from(self.slice);
        let slice: &'a mut [T] = unsafe { std::mem::transmute(vec.as_mut_slice()) };
         Tensor {
            shape: self.shape,
            vec: Some(vec),
            slice,
         }
    }
}

#[derive(Debug)]
pub struct TensorMut<'a, T, const DIM: usize> {
    pub shape: [usize; DIM],
    _vec: Option<Vec<T>>,
    pub slice: &'a mut [T],
}

impl<'a, T, const DIM: usize> D<DIM> for TensorMut<'a, T, DIM> {
    fn shape(&self) -> [usize; DIM] {
        self.shape
    }
}

impl<'a, T, const DIM: usize> TensorMut<'a, T, DIM>
where
    T: SampleUniform + PartialOrd,
    Range<T>: Clone,
{
    pub fn rand(shape: [usize; DIM], range: Range<T>) -> Self {
        let size = shape.iter().fold(1, |a, b| a * b);
        let mut rng = rand::thread_rng();
        let mut vec: Vec<T> = Vec::with_capacity(size);
        for _ in 0..size {
            vec.push(rng.gen_range(range.clone())); // Surprisingly ranges are iterators so need cloning
        }
        let slice: &'a mut [T] = unsafe { std::mem::transmute(vec.as_mut_slice()) };
        TensorMut {
            shape,
            _vec: Some(vec),
            slice,
        }
    }
}

// Vector
pub type Vector<'a, T> = Tensor<'a, T, 1>;
pub type VectorMut<'a, T> = TensorMut<'a, T, 1>;

impl<'a, T> VectorMut<'a, T>
where
    T: Copy + Default,
{
    pub fn new(i: usize) -> Self {
        let mut vec = vec![T::default(); i];
        let slice: &'a mut [T] = unsafe { std::mem::transmute(vec.as_mut_slice()) };
        VectorMut {
            shape: [i],
            _vec: Some(vec),
            slice,
        }
    }

    pub fn new_from(y: Vector<'a, T>) -> Self {
        let [size] = y.shape;
        let mut vec = vec![T::default(); size];
        let slice: &'a mut [T] = unsafe { std::mem::transmute(vec.as_mut_slice()) };
        for i in 0..size {
            vec[i] = y[i];
        }
        VectorMut {
            shape: [size],
            _vec: Some(vec),
            slice,
        }
    }
}

// Tensor2Mut
pub type Tensor2Mut<'a, T> = TensorMut<'a, T, 2>;

impl<'a, T> Tensor2Mut<'a, T>
where
    T: Copy + Default,
{
    pub fn new(i: usize, j: usize) -> Self {
        let mut vec = vec![T::default(); i * j];
        let slice: &'a mut [T] = unsafe { std::mem::transmute(vec.as_mut_slice()) };
        Tensor2Mut {
            shape: [i, j],
            _vec: Some(vec),
            slice,
        }
    }
}

// Tensor3Mut
pub type Tensor3Mut<'a, T> = TensorMut<'a, T, 3>;

impl<'a, T> Tensor3Mut<'a, T>
where
    T: Copy + Default,
{
    pub fn new(i: usize, j: usize, k: usize) -> Self {
        let mut vec = vec![T::default(); i * j * k];
        let slice: &'a mut [T] = unsafe { std::mem::transmute(vec.as_mut_slice()) };
        Tensor3Mut {
            shape: [i, j, k],
            _vec: Some(vec),
            slice,
        }
    }
}

pub type Tensor2<'a, T> = Tensor<'a, T, 2>;
pub trait Tensorify2<'a, T> {
    fn to_tensor2(&self, clone: bool) -> Tensor2<T>;
}

pub trait Vectorify<'a, T> {
    fn to_vector(&self, clone: bool) -> Vector<T>;
}

pub trait VectorifyMut<'a, T> {
    fn to_vector(&mut self) -> VectorMut<T>;
}

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

pub trait D1Get<T>: D<1> + Index<usize, Output = T> {}
pub trait D1Set<T>: D<1> + IndexMut<usize, Output = T> {}

pub trait D2Get<T>: D<2> + Index<(usize, usize), Output = T> {}

pub trait D2Set<T>: D<2> + IndexMut<(usize, usize), Output = T> {}

impl<'a, T> Index<usize> for Tensor<'a, T, 1> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &self.slice.get_unchecked(index) }
    }
}

impl<'a, T> Index<usize> for TensorMut<'a, T, 1> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &self.slice.get_unchecked(index) }
    }
}

impl<'a, T> IndexMut<usize> for TensorMut<'a, T, 1> {
    // Output defined in Index trait, T
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.slice.get_unchecked_mut(index) }
    }
}

impl<'a, T> D2Set<T> for TensorMut<'a, T, 2> {}
impl<'a, T> D2Get<T> for TensorMut<'a, T, 2> {}
impl<'a, T> D2Get<T> for Tensor<'a, T, 2> {}

impl<'a, T> D1Set<T> for VectorMut<'a, T> {}
impl<'a, T> D1Get<T> for VectorMut<'a, T> {}
impl<'a, T> D1Get<T> for Vector<'a, T> {}

impl<'a, T> Index<(usize, usize)> for Tensor<'a, T, 2> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        let [d0, d1] = self.shape;
        debug_assert!(i < d1 && j < d0);
        unsafe { &self.slice.get_unchecked(i * d0 + j) }
    }
}

impl<'a, T> Index<(usize, usize)> for TensorMut<'a, T, 2> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        let [d0, d1] = self.shape;
        debug_assert!(i < d1 && j < d0);
        unsafe { &self.slice.get_unchecked(i * d0 + j) }
    }
}

impl<'a, T> IndexMut<(usize, usize)> for TensorMut<'a, T, 2> {
    // Output defined in Index trait, T
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        let [d0, d1] = self.shape;
        debug_assert!(i < d1 && j < d0);
        unsafe { self.slice.get_unchecked_mut(i * d0 + j) }
        //&mut self.slice[i * d0 + j]
    }
}

impl<'a, T: Copy> Tensor2<'a, T> {
    pub fn row(&self, i: usize) -> Vector<'a, T> {
        let [d0, d1] = self.shape;
        debug_assert!(i < d1);
        Vector {
            shape: [d0],
            vec: None,
            slice: &self.slice[i * d0..(i + 1) * d0],
        }
    }
}

impl<'a, T: Copy> Tensor2Mut<'a, T> {
    pub fn row(&mut self, i: usize) -> VectorMut<T> {
        let [_, d1] = self.shape;
        debug_assert!(i < d1);
        VectorMut {
            shape: [d1],
            _vec: None,
            slice: &mut self.slice[i * d1..(i + 1) * d1],
        }
    }
}

pub fn softmax(v: &mut (impl D1Set<f32> + D1Get<f32>), size: usize) {
    let [d0] = v.shape();
    debug_assert!(size < d0);
    let mut max = v[0];
    for i in 1..size {
        if v[i] > max {
            max = v[i]
        }
    }
    let mut sum = 0.0;
    for i in 0..size {
        let val = f32::exp(v[i] - max);
        v[i] = val;
        sum += val;
    }
    for i in 0..size {
        v[i] = v[i] / sum;
    }
}

pub unsafe fn matmul<'a, 'b, T>(
    v1: &mut impl D1Set<T>,
    m0: &'a impl D2Get<T>,
    v0: &'b impl D1Get<T>,
) where
    T: Mul<T, Output = T>,
    T: 'a + 'b,
    T: Add<T, Output = T> + Default + Copy,
{
    let [m0d0, m0d1] = m0.shape();
    let [v0d0] = v0.shape();
    let [v1d0] = v1.shape();
    debug_assert!(m0d0 == v0d0);
    debug_assert!(m0d1 == v1d0);
    for i in 0..m0d1 {
        let mut r: T = T::default();
        for j in 0..m0d0 {
            r = r + m0[(i, j)] * v0[j];
        }
        v1[i] = r;
    }
}

pub fn rmsnorm(
    xout: &mut impl D1Set<f32>,
    xin: &impl D1Get<f32>,
    w: &impl D1Get<f32>,
    epsilon: f32,
) {
    let size = xin.shape()[0];
    let mut ss = 0.0;
    for i in 0..size {
        ss = ss + xin[i] * xin[i];
    }
    ss /= size as f32;
    ss += epsilon;
    ss = 1.0 / ss.sqrt();
    for i in 0..size {
        xout[i] = w[i] * (ss * xin[i]);
    }
}

pub fn acc(x: &mut (impl D1Set<f32> + D1Get<f32>), y: &impl D1Get<f32>) {
    let [d0] = x.shape();
    debug_assert!(d0 == y.shape()[0]);
    for i in 0..d0 {
        x[i] = x[i] + y[i];
    }
}

pub fn cp(x: &mut impl D1Set<f32>, y: &impl D1Get<f32>) {
    let [d0] = x.shape();
    debug_assert!(d0 == y.shape()[0]);
    for i in 0..d0 {
        x[i] = y[i];
    }
}

pub fn max<T>(x: &impl D1Get<T>) -> (usize, T)
where
    T: PartialOrd + Copy,
{
    let mut i = 0;
    let mut m = x[0];
    for j in 1..x.shape()[0] {
        if x[j] > m {
            i = j;
            m = x[j];
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
