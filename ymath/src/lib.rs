use core::ops::{Add, Div, Mul, Neg, Sub};
use half::f16;
use memmap2::Mmap;
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;
use std::ops::{Index, IndexMut};
use std::rc::Rc;

// Float type
pub trait Float<T>:
    Copy
    + Add<T, Output = T>
    + Sub<T, Output = T>
    + Mul<T, Output = T>
    + Div<T, Output = T>
    + Neg<Output = T>
    + Into<f32>
    + PartialOrd
{
    fn zero() -> T;
    fn one() -> T;
    fn from_f32(x: f32) -> T;
    fn to_f32(self) -> f32;
    fn from_usize(x: usize) -> T;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    fn powf(self, x: T) -> T;
}

impl Float<f32> for f32 {
    fn zero() -> f32 {
        0.0
    }
    fn one() -> f32 {
        1.0
    }
    fn from_f32(x: f32) -> f32 {
        x
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn from_usize(x: usize) -> f32 {
        x as f32
    }
    fn exp(self) -> f32 {
        f32::exp(self)
    }
    fn sqrt(self) -> f32 {
        f32::sqrt(self)
    }
    fn cos(self) -> f32 {
        f32::cos(self)
    }
    fn sin(self) -> f32 {
        f32::sin(self)
    }
    fn powf(self, x: f32) -> f32 {
        f32::powf(self, x)
    }
}

impl Float<f16> for f16 {
    fn zero() -> f16 {
        f16::from_f32(0.0)
    }
    fn one() -> f16 {
        f16::from_f32(1.0)
    }
    fn from_f32(x: f32) -> f16 {
        f16::from_f32(x)
    }
    fn to_f32(self) -> f32 {
        self.into()
    }
    fn from_usize(x: usize) -> f16 {
        f16::from_f32(x as f32)
    }
    fn exp(self) -> f16 {
        f16::from_f32(f32::exp(self.to_f32()))
    }
    fn sqrt(self) -> f16 {
        f16::from_f32(f32::sqrt(self.to_f32()))
    }
    fn cos(self) -> f16 {
        f16::from_f32(f32::cos(self.to_f32()))
    }
    fn sin(self) -> f16 {
        f16::from_f32(f32::sin(self.to_f32()))
    }
    fn powf(self, x: f16) -> f16 {
        f16::from_f32(f32::powf(self.to_f32(), x.to_f32()))
    }
}

// Tensor traits
pub trait D<const DIM: usize> {
    fn shape(&self) -> [usize; DIM];
}

pub struct MmapStore<U> {
    phantom: PhantomData<U>,
}
pub struct SubStore<U> {
    phantom: PhantomData<U>,
}

pub trait TensorTypes<T, const DIM: usize> {
    type StoreType<'a>
    where
        T: 'a;
    type ReaderType<'a>
    where
        T: 'a;
    type WriterType<'a>
    where
        T: 'a;
    type Shape;
}

impl<T, const DIM: usize> TensorTypes<T, DIM> for MmapStore<T> {
    type Shape = [usize; DIM];
    type StoreType<'a> = (Rc<Mmap>, &'a [T]) where T: 'a;
    type ReaderType<'a> = (Self::Shape, &'a [T]) where T: 'a;
    type WriterType<'a> = (Self::Shape, &'a mut [T]) where T: 'a;
}

impl<T, const DIM: usize> TensorTypes<T, DIM> for SubStore<T> {
    type Shape = [usize; DIM];
    type StoreType<'a> = &'a [T] where T: 'a;
    type ReaderType<'a> = (Self::Shape, &'a [T]) where T: 'a;
    type WriterType<'a> = (Self::Shape, &'a mut [T]) where T: 'a;
}

impl<const DIM: usize> TensorTypes<f32, DIM> for MmapStore<f16> {
    type Shape = [usize; DIM];
    type StoreType<'a> = (Rc<Mmap>, &'a [f16]);
    type ReaderType<'a> = (Self::Shape, Vec<f32>);
    type WriterType<'a> = (Self::Shape, &'a mut [f16]);
}
// Tensor /////////////////////////////////////////////////////////////////////
pub struct Tensor<'a, T: 'a, const DIM: usize, U: TensorTypes<T, DIM>> {
    pub shape: [usize; DIM],
    pub store: U::StoreType<'a>,
}

pub trait TReader<T, const DIM: usize> {
    type Reader<'b>: TRead<T, DIM>
    where
        Self: 'b;
    fn reader<'a>(&'a self) -> Self::Reader<'a>;
}

pub trait TWriter<T: Copy, const DIM: usize> {
    type Writer<'b>: TWrite<T, DIM>
    where
        Self: 'b;
    fn writer<'a>(&'a mut self) -> Self::Writer<'a>;
}

pub trait TRead<T, const DIM: usize> {
    fn reading(&self) -> ([usize; DIM], &[T]);
}

impl<'a, T, const DIM: usize> TReader<T, DIM> for Tensor<'a, T, DIM, MmapStore<T>> {
    type Reader<'b> = ([usize; DIM], &'b [T]) where Self: 'b;
    fn reader<'c>(&'c self) -> Self::Reader<'c> {
        (self.shape, &self.store.1)
    }
}

impl<'a, const DIM: usize> TReader<f32, DIM> for Tensor<'a, f32, DIM, MmapStore<f16>>
where MmapStore<f16>: TensorTypes<f16, DIM> {
    type Reader<'b> = ([usize; DIM], Vec<f32>) where Self: 'b;
    fn reader<'c>(&'c self) -> Self::Reader<'c> {
        let slice = self.store.1.iter().map(|&value| f32::from(value)).collect();
        (self.shape, slice)
    }
}


impl<'a, T, const DIM: usize> TReader<T, DIM> for Tensor<'a, T, DIM, SubStore<T>> {
    type Reader<'b> = ([usize; DIM], &'b [T]) where Self: 'b;
    fn reader(&self) -> Self::Reader<'a> {
        (self.shape, &self.store)
    }
}

impl<'a, 'b, T, const DIM: usize> TRead<T, DIM> for ([usize; DIM], &'b [T]) {
    fn reading(&self) -> ([usize; DIM], &[T]) {
        (self.0, self.1)
    }
}

impl<'a, 'b, T, const DIM: usize> TRead<T, DIM> for ([usize; DIM], Vec<T>) {
    fn reading(&self) -> ([usize; DIM], &[T]) {
        (self.0, self.1.as_slice())
    }
}

impl<'a, T: From<U>, U, const DIM: usize> D<DIM> for Tensor<'a, T, DIM, U>
where
    U: TensorTypes<T, DIM>,
{
    fn shape(&self) -> [usize; DIM] {
        self.shape
    }
}

pub trait TWrite<T: Copy, const DIM: usize> {
    fn writing(&mut self) -> ([usize; DIM], &mut [T]);
}

impl<'a: 'b, 'b, T: Copy, const DIM: usize> TWrite<T, DIM> for ([usize; DIM], &'b mut [T]) {
    fn writing(&mut self) -> ([usize; DIM], &mut [T]) {
        (self.0, self.1)
    }
}

// TensorMut
#[derive(Debug)]
pub struct TensorMut<'a, T, const DIM: usize> {
    pub shape: [usize; DIM],
    vec: Option<Vec<T>>,
    pub slice: &'a mut [T],
}

impl<'a, T: Copy, const DIM: usize> TReader<T, DIM> for TensorMut<'a, T, DIM> {
    type Reader<'b> = ([usize; DIM], &'b [T]) where Self: 'b;
    fn reader(&self) -> ([usize; DIM], &[T]) {
        (self.shape, self.slice)
    }
}

impl<'a, T: Copy, const DIM: usize> TWriter<T, DIM> for TensorMut<'a, T, DIM> {
    type Writer<'b> = ([usize; DIM], &'b mut [T]) where Self: 'b;
    fn writer(&mut self) -> ([usize; DIM], &mut [T]) {
        (self.shape, self.slice)
    }
}

impl<'a, T, const DIM: usize> TRead<T, DIM> for TensorMut<'a, T, DIM> {
    fn reading(&self) -> ([usize; DIM], &[T]) {
        (self.shape, self.slice)
    }
}

impl<'a, T: Copy, const DIM: usize> TWrite<T, DIM> for TensorMut<'_, T, DIM> {
    fn writing(&mut self) -> ([usize; DIM], &mut [T]) {
        (self.shape, self.slice)
    }
}

impl<'a, T, const DIM: usize> D<DIM> for TensorMut<'a, T, DIM> {
    fn shape(&self) -> [usize; DIM] {
        self.shape
    }
}

impl<'a, T, const DIM: usize> Clone for TensorMut<'a, T, DIM>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        let mut vec = self.vec.clone();
        let slice = unsafe { std::mem::transmute(vec.as_mut_slice()) };
        TensorMut {
            shape: self.shape.clone(),
            vec,
            slice,
        }
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
            vec: Some(vec),
            slice,
        }
    }
}

// Vector
pub type Vector<'a, T, U> = Tensor<'a, T, 1, U>;
pub type VectorMut<'a, T> = TensorMut<'a, T, 1>;

impl<'a, T: Float<T>> VectorMut<'a, T> {
    pub fn new(i: usize) -> Self {
        let mut vec = vec![T::zero(); i];
        let slice: &'a mut [T] = unsafe { std::mem::transmute(vec.as_mut_slice()) };
        VectorMut {
            shape: [i],
            vec: Some(vec),
            slice,
        }
    }

    pub fn new_from(y: &'a mut Vector<'a, T, SubStore<T>>) -> Self {
        let ([size], y_slice) = y.reader();
        let mut vec = vec![T::zero(); size];
        let slice: &'a mut [T] = unsafe { std::mem::transmute(vec.as_mut_slice()) };
        for i in 0..size {
            vec[i] = y_slice[i];
        }
        VectorMut {
            shape: [size],
            vec: Some(vec),
            slice,
        }
    }
}

// Tensor2Mut
pub type Tensor2Mut<'a, T> = TensorMut<'a, T, 2>;

impl<'a, T: Float<T>> Tensor2Mut<'a, T> {
    pub fn new(i: usize, j: usize) -> Self {
        let mut vec = vec![T::zero(); i * j];
        let slice: &'a mut [T] = unsafe { std::mem::transmute(vec.as_mut_slice()) };
        Tensor2Mut {
            shape: [i, j],
            vec: Some(vec),
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
            vec: Some(vec),
            slice,
        }
    }
}

// Tensorify
pub type Tensor2<'a, T, E> = Tensor<'a, T, 2, E>;

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

impl<'a, T> Index<usize> for Tensor<'a, T, 1, MmapStore<T>> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        let reader = self.reader();
        let slice = reader.reading().1;
        unsafe {
            let slice: &'a [T] = std::mem::transmute(slice);
            slice.get_unchecked(index)
        }
    }
}

impl<'a, T> Index<usize> for Tensor<'a, T, 1, SubStore<T>> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        let reader = self.reader();
        let slice = reader.reading().1;
        unsafe {
            let slice: &'a [T] = std::mem::transmute(slice);
            slice.get_unchecked(index)
        }
    }
}

impl<'a, T> Index<usize> for TensorMut<'a, T, 1> {
    type Output = T;
    fn index<'b>(&'b self, index: usize) -> &'b Self::Output {
        let (_, slice) = self.reading();
        unsafe { slice.get_unchecked(index) }
    }
}

impl<'a, T: Copy> IndexMut<usize> for TensorMut<'a, T, 1> {
    // Output defined in Index trait, T
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let (_, slice) = self.writing();
        unsafe { slice.get_unchecked_mut(index) }
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

pub trait Rowable<'a, T, U>
where
    U: TensorTypes<T, 1> + TensorTypes<T, 1>,
{
    fn row(&self, i: usize) -> Vector<'a, T, U>;
}

impl<'a, T: Copy> Rowable<'a, T, MmapStore<T>> for Tensor<'a, T, 2, MmapStore<T>> {
    fn row(&self, i: usize) -> Vector<'a, T, MmapStore<T>> {
        let ([d0, d1], _) = self.reader().reading();
        debug_assert!(i < d1);
        Vector {
            shape: [d0],
            store: (self.store.0.clone(), &self.store.1[i * d0..(i + 1) * d0]),
        }
    }
}

impl<'a> Rowable<'a, f32, MmapStore<f16>> for Tensor<'a, f32, 2, MmapStore<f16>> {
    fn row(&self, i: usize) -> Vector<'a, f32, MmapStore<f16>> {
        let ([d0, d1], _) = self.reader().reading();
        debug_assert!(i < d1);
        Vector {
            shape: [d0],
            store: (self.store.0.clone(), &self.store.1[i * d0..(i + 1) * d0]),
        }
    }
}

impl<'a, T: Copy> Tensor2Mut<'a, T> {
    pub fn row(&mut self, i: usize) -> VectorMut<T> {
        let ([_, d1], slice) = self.writing();
        debug_assert!(i < d1);
        VectorMut {
            shape: [d1],
            vec: None,
            slice: &mut slice[i * d1..(i + 1) * d1],
        }
    }
}

pub fn softmax<T: Copy + Float<T>>(v: &mut impl TWrite<T, 1>, size: usize) {
    let ([d0], v) = v.writing();
    debug_assert!(size < d0);
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

pub unsafe fn matmul<T>(
    v1: &mut impl TWriter<T, 1>,
    m0: &mut impl TReader<T, 2>,
    v0: &mut impl TReader<T, 1>,
) where
    T: Float<T>,
{
    let m0_reader = m0.reader();
    let ([m0d0, m0d1], m0_slice) = m0_reader.reading();
    let v0_reader = v0.reader();
    let ([v0d0], v0_slice) = v0_reader.reading();
    let mut v1_writer = v1.writer();
    let ([v1d0], v1_slice) = v1_writer.writing();
    debug_assert!(m0d0 == v0d0);
    debug_assert!(m0d1 == v1d0);
    for i in 0..m0d1 {
        let mut r = T::zero();
        for j in 0..m0d0 {
            r = r + m0_slice[i * m0d0 + j] * v0_slice[j];
        }
        v1_slice[i] = r;
    }
}

pub fn rmsnorm<'a, 'b, T: Float<T>>(
    xout: &'a mut impl TWriter<T, 1>,
    xin: &'a mut impl TReader<T, 1>,
    w: &'a mut impl TReader<T, 1>,
    epsilon: T,
) {
    let xin = xin.reader();
    let ([xind], xins) = xin.reading();
    let w = w.reader();
    let ([_], ws) = w.reading();
    let mut xout = xout.writer();
    let ([_], xouts) = xout.writing();
    let size = xind;
    let mut ss = T::zero();
    for i in 0..size {
        ss = ss + xins[i] * xins[i];
    }
    ss = ss / T::from_usize(size) + epsilon;
    ss = T::one() / ss.sqrt();
    for i in 0..size {
        xouts[i] = ws[i] * (ss * xins[i]);
    }
}

pub fn acc<'a, T>(x: &'a mut impl TWriter<T, 1>, y: &'a mut impl TReader<T, 1>)
where
    T: Copy + Add<T, Output = T> + 'a,
{
    let mut x = x.writer();
    let ([xd], xs) = x.writing();
    let y = y.reader();
    let ([yd], ys) = y.reading();
    debug_assert!(xd == yd);
    for i in 0..xd {
        xs[i] = xs[i] + ys[i];
    }
}

pub fn cp<'a, T: Copy + 'a>(x: &'a mut impl TWriter<T, 1>, y: &'a mut impl TReader<T, 1>) {
    let y = y.reader();
    let ([yd], ys) = y.reading();
    let mut x = x.writer();
    let ([xd], xs) = x.writing();
    debug_assert!(xd == yd);
    for i in 0..xd {
        xs[i] = ys[i];
    }
}

pub fn max<'a, T: 'a>(x: &'a mut impl TRead<T, 1>) -> (usize, T)
where
    T: PartialOrd + Copy,
{
    let ([size], slice) = x.reading();
    let mut i = 0;
    let mut m = slice[0];
    for j in 1..size {
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
