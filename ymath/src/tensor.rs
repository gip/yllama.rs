use crate::*;
use std::marker::PhantomData;

// Shape //////////////////////////////////////////////////////////////////////
pub trait IsTensor {
    fn n_elem() -> usize;
    fn dim() -> usize;
}

pub trait IsMatrix: IsTensor {}
pub trait IsVector: IsTensor {}

pub struct VECTOR<const D0: usize> {}
pub type V<const D0: usize> = VECTOR<D0>;
impl<const D0: usize> IsTensor for VECTOR<D0> {
    fn n_elem() -> usize {
        D0
    }
    fn dim() -> usize {
        1
    }
}
impl<const D0: usize> IsVector for VECTOR<D0> {}

pub struct MATRIX<const D0: usize, const D1: usize> {}
pub type M<const D0: usize, const D1: usize> = MATRIX<D0, D1>;
impl<const D0: usize, const D1: usize> IsTensor for MATRIX<D0, D1> {
    fn n_elem() -> usize {
        D0 * D1
    }
    fn dim() -> usize {
        2
    }
}
impl<const D0: usize, const D1: usize> IsVector for MATRIX<D0, D1> {}

// Tensor /////////////////////////////////////////////////////////////////////
pub trait TensorTypes<T, SHAPE> {
    type StoreType<'a>
    where
        T: 'a;
    type ReaderType<'a>
    where
        T: 'a;
    type WriterType<'a>
    where
        T: 'a;
}

pub trait TReader<T, SHAPE: IsTensor> {
    type Reader<'b>: TRead<T>
    where
        Self: 'b;
    fn reader<'a>(&'a self) -> Self::Reader<'a>;
}

pub trait TRead<T> {
    fn reading(&self) -> &[T];
}

pub trait TWriter<T: Copy, SHAPE: IsTensor>: TReader<T, SHAPE> {
    type Writer<'b>: TWrite<T>
    where
        Self: 'b;
    fn writer<'a>(&'a mut self) -> Self::Writer<'a>;
}

pub trait TWrite<T: Copy> {
    fn writing(&mut self) -> &mut [T];
}

// TensorMut //////////////////////////////////////////////////////////////////
type TensorMut<'a, T, SHAPE, U = VecStore<T>> = Tensor<'a, true, T, SHAPE, U>;

pub type VectorMut<'a, T, const D0: usize> = TensorMut<'a, T, VECTOR<D0>>;
pub type Tensor2Mut<'a, T, const D0: usize, const D1: usize> = TensorMut<'a, T, MATRIX<D0, D1>>;

impl<'a, T, SHAPE: IsTensor> TReader<T, SHAPE> for TensorMut<'a, T, SHAPE> {
    type Reader<'b> = &'b [T] where Self: 'b;
    fn reader(&self) -> &[T] {
        self.store.as_slice()
    }
}

impl<'a, T: Copy, SHAPE: IsTensor> TWriter<T, SHAPE> for TensorMut<'a, T, SHAPE> {
    type Writer<'b> = &'b mut [T] where Self: 'b;
    fn writer(&mut self) -> &mut [T] {
        self.store.as_mut_slice()
    }
}

impl<'a, T: Float, const D0: usize> VectorMut<'a, T, D0> {
    pub fn new() -> Self {
        let vec = vec![T::zero(); D0];
        VectorMut { store: vec }
    }
}

impl<'a, T: Float, const D0: usize, const D1: usize> Tensor2Mut<'a, T, D0, D1> {
    pub fn new() -> Self {
        let vec = vec![T::zero(); D0 * D1];
        TensorMut { store: vec }
    }
}

impl<'a, T, const D0: usize> Index<usize> for TensorMut<'a, T, VECTOR<D0>> {
    type Output = T;
    fn index<'b>(&'b self, index: usize) -> &'b Self::Output {
        let reader = self.reader();
        let slice = reader.reading();
        #[cfg(debug_assertions)]
        unsafe {
            std::mem::transmute(slice.get_unchecked(index))
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            std::mem::transmute(slice.get(index))
        }
    }
}

impl<'a, T, const D0: usize> Index<usize> for TensorMut<'a, T, VECTOR<D0>, SubStore<T, true>> {
    type Output = T;
    fn index<'b>(&'b self, index: usize) -> &'b Self::Output {
        let reader = self.reader();
        let slice = reader.reading();
        #[cfg(debug_assertions)]
        unsafe {
            std::mem::transmute(slice.get_unchecked(index))
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            std::mem::transmute(slice.get(index))
        }
    }
}

impl<'a, T, const D0: usize> Index<usize> for TensorMut<'a, T, VECTOR<D0>, SubStore<T, false>> {
    type Output = T;
    fn index<'b>(&'b self, index: usize) -> &'b Self::Output {
        let reader = self.reader();
        let slice = reader.reading();
        #[cfg(debug_assertions)]
        unsafe {
            std::mem::transmute(slice.get_unchecked(index))
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            std::mem::transmute(slice.get(index))
        }
    }
}

impl<'a, T: Copy, const D0: usize> IndexMut<usize> for TensorMut<'a, T, VECTOR<D0>> {
    // Output defined in Index trait, T
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let mut writer = self.writer();
        let slice = writer.writing();
        #[cfg(debug_assertions)]
        unsafe {
            std::mem::transmute(slice.get_unchecked_mut(index))
        }
        #[cfg(not(debug_assertions))]
        {
            unsafe { std::mem::transmute(slice.get_mut(index)) }
        }
    }
}

impl<'a, T: Copy, const D0: usize> IndexMut<usize>
    for TensorMut<'a, T, VECTOR<D0>, SubStore<T, true>>
{
    // Output defined in Index trait, T
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let mut writer = self.writer();
        let slice = writer.writing();
        #[cfg(debug_assertions)]
        unsafe {
            std::mem::transmute(slice.get_unchecked_mut(index))
        }
        #[cfg(not(debug_assertions))]
        {
            unsafe { std::mem::transmute(slice.get_mut(index)) }
        }
    }
}

impl<'a, T, const D0: usize, const D1: usize> Index<(usize, usize)>
    for TensorMut<'a, T, MATRIX<D0, D1>>
{
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        debug_assert!(i < D1 && j < D0);
        unsafe { &self.store.as_slice().get_unchecked(i * D0 + j) }
    }
}

impl<'a, T, const D0: usize, const D1: usize> IndexMut<(usize, usize)>
    for TensorMut<'a, T, MATRIX<D0, D1>>
{
    // Output defined in Index trait, T
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        debug_assert!(i < D1 && j < D0);
        unsafe { self.store.as_mut_slice().get_unchecked_mut(i * D0 + j) }
        //&mut self.slice[i * d0 + j]
    }
}

impl<'a, T, const D0: usize, const D1: usize> RowableMut<T, D0, D1, VecStore<T>>
    for Tensor<'a, true, T, MATRIX<D0, D1>, VecStore<T>>
where
    T: 'a,
{
    type RowStoreType = SubStore<T>;
    type RowTensorType<'b> = Tensor<'b, true, T, VECTOR<D0>, SubStore<T, true>> where Self: 'b;

    fn row<'b>(&'b mut self, i: usize) -> Self::RowTensorType<'b> {
        debug_assert!(i < D1);
        Tensor {
            store: &mut self.store[i * D0..(i + 1) * D0],
        }
    }
}

// Tensor /////////////////////////////////////////////////////////////////////
pub struct Tensor<'a, const RW: bool, T: 'a, SHAPE: IsTensor, U: TensorTypes<T, SHAPE>> {
    pub store: U::StoreType<'a>,
}

pub type VectorImm<'a, T, const D0: usize, U> = Tensor<'a, false, T, VECTOR<D0>, U>;
pub type Tensor2Imm<'a, T, const D0: usize, const D1: usize, U> =
    Tensor<'a, false, T, MATRIX<D0, D1>, U>;

// MmapStore //////////////////////////////////////////////////////////////////
pub struct MmapStore<T, U, const CP: bool = true> {
    phantom_t: PhantomData<T>,
    phantom_u: PhantomData<U>,
}

// Strategy: data on mmap, no copy
impl<T, SHAPE: IsTensor> TensorTypes<T, SHAPE> for MmapStore<T, T> {
    type StoreType<'a> = (Rc<Mmap>, &'a [T]) where T: 'a;
    type ReaderType<'a> = &'a [T] where T: 'a;
    type WriterType<'a> = &'a mut [T] where T: 'a;
}

impl<'a, T, SHAPE: IsTensor> TReader<T, SHAPE> for Tensor<'a, false, T, SHAPE, MmapStore<T, T>> {
    type Reader<'b> = &'b [T] where Self: 'b;
    fn reader<'c>(&'c self) -> Self::Reader<'c> {
        self.store.1
    }
}

// Strategy: f16 data on mmap, full data conversion + on-demand copy to a Vec<f32>
impl<SHAPE: IsTensor> TensorTypes<f32, SHAPE> for MmapStore<f32, f16, true> {
    type StoreType<'a> = (Rc<Mmap>, &'a [f16]);
    type ReaderType<'a> = Vec<f32>;
    type WriterType<'a> = &'a mut [f16];
}

impl<'a, SHAPE: IsTensor> TReader<f32, SHAPE>
    for Tensor<'a, false, f32, SHAPE, MmapStore<f32, f16, true>>
{
    type Reader<'b> = Vec<f32> where Self: 'b;
    fn reader<'c>(&'c self) -> Self::Reader<'c> {
        self.store.1.iter().map(|&value| f32::from(value)).collect()
    }
}

// Strategy: f16 data on mmap, on the fly data conversion
impl<SHAPE: IsTensor> TensorTypes<f32, SHAPE> for MmapStore<f32, f16, false> {
    type StoreType<'a> = (Rc<Mmap>, &'a [f16]);
    type ReaderType<'a> = &'a [f16];
    type WriterType<'a> = &'a mut [f16];
}

impl<'a, T, const D0: usize, const D1: usize> Rowable<T, D0, D1, MmapStore<T, T>>
    for Tensor<'a, false, T, MATRIX<D0, D1>, MmapStore<T, T>>
where
    T: 'a,
{
    type RowStoreType = SubStore<T>;
    type RowTensorType<'b> = Tensor<'b, false, T, VECTOR<D0>, MmapStore<T, T>> where Self: 'b;

    fn row<'b>(&'b self, i: usize) -> Self::RowTensorType<'b> {
        debug_assert!(i < D1);
        Tensor {
            store: (self.store.0.clone(), &self.store.1[i * D0..(i + 1) * D0]),
        }
    }
}

impl<'a, T, SHAPE: IsVector + 'a, U> Index<usize> for Tensor<'a, false, T, SHAPE, MmapStore<T, U>>
where
    MmapStore<T, U>: TensorTypes<T, SHAPE>,
    Self: TReader<T, SHAPE>,
{
    type Output = T;
    fn index(&self, index: usize) -> &T {
        let reader = self.reader();
        let slice = reader.reading();
        #[cfg(debug_assertions)]
        unsafe {
            std::mem::transmute(slice.get_unchecked(index))
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            std::mem::transmute(slice.get(index))
        }
    }
}

// SubStore ///////////////////////////////////////////////////////////////////
pub struct SubStore<U, const RW: bool = false> {
    phantom: PhantomData<U>,
}

impl<T, SHAPE: IsTensor> TensorTypes<T, SHAPE> for SubStore<T, false> {
    type StoreType<'a> = &'a [T] where T: 'a;
    type ReaderType<'a> = &'a [T] where T: 'a;
    type WriterType<'a> = () where T: 'a;
}

impl<T, SHAPE: IsTensor> TensorTypes<T, SHAPE> for SubStore<T, true> {
    type StoreType<'a> = &'a mut [T] where T: 'a;
    type ReaderType<'a> = &'a [T] where T: 'a;
    type WriterType<'a> = &'a mut [T] where T: 'a;
}

impl<'a, T, SHAPE: IsTensor + 'a> TReader<T, SHAPE> for Tensor<'a, false, T, SHAPE, SubStore<T, false>> {
    type Reader<'b> = &'b [T] where Self: 'b;
    fn reader(&self) -> Self::Reader<'_> {
        self.store
    }
}

// impl<'a, T, SHAPE: IsTensor + 'a> TReader<T, SHAPE> for Tensor<'a, true, T, SHAPE, SubStore<T, true>> {
//     type Reader<'b> = &'b [T] where Self: 'b;
//     fn reader(&self) -> Self::Reader<'_> {
//         self.store
//     }
// }

// impl<'a, T: Copy, SHAPE: IsTensor + 'a> TWriter<T, SHAPE> for Tensor<'a, true, T, SHAPE, SubStore<T, true>> {
//     type Writer<'b> = &'b mut [T] where Self: 'b;
//     fn writer(&mut self) -> Self::Writer<'_> {
//         self.store
//     }
// }

impl<'a, T, SHAPE: IsTensor> TReader<T, SHAPE> for TensorMut<'a, T, SHAPE, SubStore<T, false>> {
    type Reader<'b> = &'b [T] where Self: 'b;
    fn reader(&self) -> &[T] {
        self.store
    }
}

impl<'a, T, SHAPE: IsTensor> TReader<T, SHAPE> for TensorMut<'a, T, SHAPE, SubStore<T, true>> {
    type Reader<'b> = &'b [T] where Self: 'b;
    fn reader(&self) -> &[T] {
        self.store
    }
}

impl<'a, T: Copy, SHAPE: IsTensor> TWriter<T, SHAPE>
    for TensorMut<'a, T, SHAPE, SubStore<T, true>>
{
    type Writer<'b> = &'b mut [T] where Self: 'b;
    fn writer(&mut self) -> &mut [T] {
        self.store
    }
}

impl<'a, T, SHAPE: IsVector + 'a> Index<usize> for Tensor<'a, false, T, SHAPE, SubStore<T>> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        let reader = self.reader();
        let slice = <&[T] as tensor::TRead<T>>::reading(&reader);
        #[cfg(debug_assertions)]
        unsafe {
            std::mem::transmute(slice.get_unchecked(index))
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            std::mem::transmute(slice.get(index))
        }
    }
}

// VecStore ///////////////////////////////////////////////////////////////////
pub struct VecStore<T> {
    phantom: PhantomData<T>,
}

impl<T, SHAPE: IsTensor> TensorTypes<T, SHAPE> for VecStore<T> {
    type StoreType<'a> = Vec<T> where T: 'a;
    type ReaderType<'a> = &'a [T] where T: 'a;
    type WriterType<'a> = &'a mut [T] where T: 'a;
}

impl<'a, T, SHAPE: IsTensor> TReader<T, SHAPE> for Tensor<'a, false, T, SHAPE, VecStore<T>> {
    type Reader<'b> = &'b [T] where Self: 'b;
    fn reader<'c>(&'c self) -> Self::Reader<'c> {
        &self.store.as_slice()
    }
}

impl<'a, T, const D0: usize, const D1: usize> Rowable<T, D0, D1, VecStore<T>>
    for Tensor<'a, false, T, MATRIX<D0, D1>, VecStore<T>>
where
    T: 'a,
{
    type RowStoreType = SubStore<T>;
    type RowTensorType<'b> = Tensor<'b, false, T, VECTOR<D0>, SubStore<T>> where Self: 'b;

    fn row<'b>(&'b self, i: usize) -> Self::RowTensorType<'b> {
        debug_assert!(i < D1);
        Tensor {
            store: &self.store[i * D0..(i + 1) * D0],
        }
    }
}

// TRead //////////////////////////////////////////////////////////////////////
impl<'a, 'b, T> TRead<T> for &'b [T] {
    fn reading(&self) -> &[T] {
        self
    }
}

impl<'a, 'b, T> TRead<T> for Vec<T> {
    fn reading(&self) -> &[T] {
        self.as_slice()
    }
}

// TWrite /////////////////////////////////////////////////////////////////////
impl<'a: 'b, 'b, T: Copy> TWrite<T> for &'b mut [T] {
    fn writing(&mut self) -> &mut [T] {
        self
    }
}

//
pub trait Rowable<T, const D0: usize, const D1: usize, S: TensorTypes<T, M<D0, D1>>> {
    type RowStoreType: TensorTypes<T, V<D0>>;
    type RowTensorType<'a>: TReader<T, V<D0>>
    where
        Self: 'a;
    fn row<'a>(&'a self, i: usize) -> Self::RowTensorType<'a>;
}
pub trait RowableMut<T, const D0: usize, const D1: usize, S: TensorTypes<T, M<D0, D1>>> {
    type RowStoreType: TensorTypes<T, V<D0>>;
    type RowTensorType<'a>: TReader<T, V<D0>>
    where
        Self: 'a;
    fn row<'a>(&'a mut self, i: usize) -> Self::RowTensorType<'a>;
}