use crate::*;
use std::cell::{Ref, RefCell, RefMut};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

// Shape traits ///////////////////////////////////////////////////////////////
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
impl<const D0: usize, const D1: usize> IsMatrix for MATRIX<D0, D1> {}

// Tensor traits //////////////////////////////////////////////////////////////
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
    type ReaderTypeA<'a>
    where
        T: 'a;
    type WriterTypeA<'a>
    where
        T: 'a;
}

pub trait Indexable {
    type IndexType;
}

impl<const D0: usize> Indexable for V<D0> {
    type IndexType = usize;
}

impl<const D0: usize, const D1: usize> Indexable for M<D0, D1> {
    type IndexType = (usize, usize);
}

pub trait TGet<Idx: ?Sized> {
    type Output;
    fn get(&self, index: Idx) -> Self::Output;
}

pub trait TSet<Idx: ?Sized>: TGet<Idx> {
    fn set(&mut self, index: Idx, val: Self::Output);
}

// TSlice
pub struct TSlice<'a, T, SHAPE: IsTensor, U = T> {
    slice: &'a [U],
    _phantom0: PhantomData<SHAPE>,
    _phantom1: PhantomData<T>,
}

pub struct TSliceMut<'a, T, SHAPE: IsTensor> {
    slice: &'a mut [T],
    _phantom: PhantomData<SHAPE>,
}

impl<'a, T: Copy, const D0: usize> TGet<usize> for TSlice<'a, T, V<D0>> {
    type Output = T;
    fn get(&self, i: usize) -> Self::Output {
        #[cfg(not(debug_assertions))]
        unsafe {
            *self.slice.get_unchecked(i)
        }
        #[cfg(debug_assertions)]
        self.slice[i]
    }
}

impl<'a, const D0: usize> TGet<usize> for TSlice<'a, f32, V<D0>, f16> {
    type Output = f32;
    fn get(&self, i: usize) -> Self::Output {
        #[cfg(not(debug_assertions))]
        unsafe {
            (*self.slice.get_unchecked(i)).into()
        }
        #[cfg(debug_assertions)]
        self.slice[i].into()
    }
}

impl<'a, T: Copy, const D0: usize, const D1: usize> TGet<(usize, usize)>
    for TSlice<'a, T, M<D0, D1>>
{
    type Output = T;
    fn get(&self, (i, j): (usize, usize)) -> Self::Output {
        #[cfg(not(debug_assertions))]
        unsafe {
            *self.slice.get_unchecked(i * D0 + j)
        }
        #[cfg(debug_assertions)]
        self.slice[i * D0 + j]
    }
}

impl<'a, T: Copy, const D0: usize> TGet<usize> for TSliceMut<'a, T, V<D0>> {
    type Output = T;
    fn get(&self, i: usize) -> Self::Output {
        #[cfg(not(debug_assertions))]
        unsafe {
            *self.slice.get_unchecked(i)
        }
        #[cfg(debug_assertions)]
        self.slice[i]
    }
}

impl<'a, T: Copy, const D0: usize, const D1: usize> TGet<(usize, usize)>
    for TSliceMut<'a, T, M<D0, D1>>
{
    type Output = T;
    fn get(&self, (i, j): (usize, usize)) -> Self::Output {
        #[cfg(not(debug_assertions))]
        unsafe {
            *self.slice.get_unchecked(i * D0 + j)
        }
        #[cfg(debug_assertions)]
        self.slice[i * D0 + j]
    }
}

impl<'a, T: Copy, const D0: usize> TSet<usize> for TSliceMut<'a, T, V<D0>> {
    fn set(&mut self, i: usize, val: T) {
        #[cfg(not(debug_assertions))]
        unsafe {
            *self.slice.get_unchecked_mut(i) = val
        }
        #[cfg(debug_assertions)]
        {
            self.slice[i] = val
        }
    }
}

impl<'a, T: Copy, const D0: usize, const D1: usize> TSet<(usize, usize)>
    for TSliceMut<'a, T, M<D0, D1>>
{
    fn set(&mut self, (i, j): (usize, usize), val: T) {
        #[cfg(not(debug_assertions))]
        unsafe {
            *self.slice.get_unchecked_mut(i * D0 + j) = val
        }
        #[cfg(debug_assertions)]
        {
            self.slice[i * D0 + j] = val
        }
    }
}

// TVec
pub struct TVec<T, SHAPE: IsTensor> {
    vec: Vec<T>,
    _phantom: PhantomData<SHAPE>,
}

impl<'a, T: Copy, const D0: usize> TGet<usize> for TVec<T, V<D0>> {
    type Output = T;
    fn get(&self, i: usize) -> Self::Output {
        self.vec[i]
    }
}

impl<'a, T: Copy, const D0: usize, const D1: usize> TGet<(usize, usize)> for TVec<T, M<D0, D1>> {
    type Output = T;
    fn get(&self, (i, j): (usize, usize)) -> Self::Output {
        self.vec[i * D0 + j]
    }
}

pub trait TReader<T, SHAPE: IsTensor + Indexable, U = T> {
    type Reader<'b>: TGet<SHAPE::IndexType, Output = T>
    where
        Self: 'b;
    fn reader<'a>(&'a self) -> Self::Reader<'a>;
}

pub trait TWriter<T: Copy, SHAPE: IsTensor + Indexable>: TReader<T, SHAPE> {
    type Writer<'b>: TSet<SHAPE::IndexType, Output = T>
    where
        Self: 'b;
    fn writer<'a>(&'a mut self) -> Self::Writer<'a>;
}

// // TCell
// pub struct TCell<T, SHAPE: IsTensor> {
//     offset: usize,
//     vec: RefCell<Vec<T>>,
//     _phantom: PhantomData<SHAPE>,
// }

// impl<'a, T: Copy, const D0: usize, const D1: usize> TGet<(usize, usize)> for TCell<T, M<D0, D1>> {
//     type Output = T;
//     fn get(&self, (i, j): (usize, usize)) -> Self::Output {
//         self.vec[i * D0 + j]
//     }
// }

// Tensor /////////////////////////////////////////////////////////////////////
pub struct Tensor<'a, const RW: bool, T: 'a, SHAPE: IsTensor, U: TensorTypes<T, SHAPE>> {
    pub store: U::StoreType<'a>,
}

pub trait Rowable<T, const D0: usize, const D1: usize, S: TensorTypes<T, M<D0, D1>>> {
    type RowStoreType: TensorTypes<T, V<D0>>;
    type RowTensorType<'a>: TReader<T, V<D0>>
    where
        Self: 'a;
    fn row<'a>(&'a self, i: usize) -> Self::RowTensorType<'a>;
}

pub trait RowableMut<T: Copy, const D0: usize, const D1: usize, S: TensorTypes<T, M<D0, D1>>> {
    type RowStoreType: TensorTypes<T, V<D0>>;
    type RowTensorType<'a>: TWriter<T, V<D0>>
    where
        Self: 'a;
    fn row<'a>(&'a mut self, i: usize) -> Self::RowTensorType<'a>;
}

pub type VectorImm<'a, T, const D0: usize, U> = Tensor<'a, false, T, VECTOR<D0>, U>;
pub type Tensor2Imm<'a, T, const D0: usize, const D1: usize, U> =
    Tensor<'a, false, T, MATRIX<D0, D1>, U>;

// TensorMut //////////////////////////////////////////////////////////////////
pub type TensorMut<'a, T, SHAPE, U = VecStore<T>> = Tensor<'a, true, T, SHAPE, U>;

pub type VectorMut<'a, T, const D0: usize> = TensorMut<'a, T, VECTOR<D0>>;
pub type Tensor2Mut<'a, T, const D0: usize, const D1: usize> = TensorMut<'a, T, MATRIX<D0, D1>>;

impl<'a, T: Copy, const D0: usize> TReader<T, V<D0>> for TensorMut<'a, T, V<D0>> {
    type Reader<'b> = TSlice<'b, T, V<D0>> where Self: 'b;
    fn reader(&self) -> Self::Reader<'_> {
        TSlice {
            slice: self.store.as_slice(),
            _phantom0: PhantomData,
            _phantom1: PhantomData,
        }
    }
}

impl<'a, T: Copy, const D0: usize, const D1: usize> TReader<T, M<D0, D1>>
    for TensorMut<'a, T, M<D0, D1>>
{
    type Reader<'b> = TSlice<'b, T, M<D0, D1>> where Self: 'b;
    fn reader(&self) -> Self::Reader<'_> {
        TSlice {
            slice: self.store.as_slice(),
            _phantom0: PhantomData,
            _phantom1: PhantomData,
        }
    }
}

impl<'a, T: Copy, const D0: usize> TWriter<T, V<D0>> for TensorMut<'a, T, V<D0>> {
    type Writer<'b> = TSliceMut<'b, T, V<D0>> where Self: 'b;
    fn writer(&mut self) -> Self::Writer<'_> {
        TSliceMut {
            slice: self.store.as_mut_slice(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Copy, const D0: usize, const D1: usize> TWriter<T, M<D0, D1>>
    for TensorMut<'a, T, M<D0, D1>>
{
    type Writer<'b> = TSliceMut<'b, T, M<D0, D1>> where Self: 'b;
    fn writer(&mut self) -> Self::Writer<'_> {
        TSliceMut {
            slice: self.store.as_mut_slice(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Float, const D0: usize> VectorMut<'a, T, D0> {
    pub fn new_vector() -> Self {
        let vec = vec![T::zero(); D0];
        VectorMut { store: vec }
    }
}

impl<'a, T: Float, const D0: usize, const D1: usize> Tensor2Mut<'a, T, D0, D1> {
    pub fn new_matrix() -> Self {
        let vec = vec![T::zero(); D0 * D1];
        TensorMut { store: vec }
    }
}

impl<'a, T: Copy, const D0: usize, const D1: usize> RowableMut<T, D0, D1, VecStore<T>>
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
    type ReaderTypeA<'a> = TSlice<'a, T, SHAPE> where T: 'a;
    type WriterTypeA<'a> = TSliceMut<'a, T, SHAPE> where T: 'a;
}

impl<'a, T, SHAPE: IsTensor + Indexable> TReader<T, SHAPE>
    for Tensor<'a, false, T, SHAPE, MmapStore<T, T>>
where
    for<'c> TSlice<'c, T, SHAPE>: TGet<<SHAPE as Indexable>::IndexType, Output = T>,
{
    type Reader<'b> = TSlice<'b, T, SHAPE> where Self: 'b;
    fn reader(&self) -> Self::Reader<'_> {
        TSlice {
            slice: self.store.1,
            _phantom0: PhantomData,
            _phantom1: PhantomData,
        }
    }
}

// Strategy: f16 data on mmap, full data conversion + on-demand copy to a Vec<f32>
impl<SHAPE: IsTensor> TensorTypes<f32, SHAPE> for MmapStore<f32, f16, true> {
    type StoreType<'a> = (Rc<Mmap>, &'a [f16]);
    type ReaderType<'a> = Vec<f32>;
    type WriterType<'a> = &'a mut [f16];
    type ReaderTypeA<'a> = TVec<f32, SHAPE>;
    type WriterTypeA<'a> = &'a mut [f16];
}

impl<'a, SHAPE: IsTensor + Indexable> TReader<f32, SHAPE>
    for Tensor<'a, false, f32, SHAPE, MmapStore<f32, f16, true>>
where
    for<'c> TVec<f32, SHAPE>: TGet<<SHAPE as Indexable>::IndexType, Output = f32>,
{
    type Reader<'b> = TVec<f32, SHAPE> where Self: 'b;
    fn reader(&self) -> Self::Reader<'_> {
        TVec {
            vec: self.store.1.iter().map(|&value| f32::from(value)).collect(),
            _phantom: PhantomData,
        }
    }
}

// Strategy: f16 data on mmap, on the fly data conversion
impl<SHAPE: IsTensor> TensorTypes<f32, SHAPE> for MmapStore<f32, f16, false> {
    type StoreType<'a> = (Rc<Mmap>, &'a [f16]);
    type ReaderType<'a> = &'a [f16];
    type WriterType<'a> = &'a mut [f16];
    type ReaderTypeA<'a> = TSlice<'a, f16, SHAPE>;
    type WriterTypeA<'a> = TSliceMut<'a, f16, SHAPE>;
}

impl<'a, T: Copy, const D0: usize, const D1: usize> Rowable<T, D0, D1, MmapStore<T, T>>
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

// RefStore ///////////////////////////////////////////////////////////////////
pub struct RefStore<'a, T> {
    #[allow(dead_code)]
    store: (usize, &'a RefCell<Vec<T>>),
    phantom: PhantomData<T>,
}

impl<'b, T, SHAPE: IsTensor> TensorTypes<T, SHAPE> for RefStore<'b, T> {
    type StoreType<'a> = (usize, &'a RefCell<Vec<T>>) where T: 'a;
    type ReaderType<'a> =() where T: 'a;
    type WriterType<'a> = () where T: 'a;
    type ReaderTypeA<'a> = TSlice<'a, T, SHAPE> where T: 'a;
    type WriterTypeA<'a> = TSliceMut<'a, T, SHAPE> where T: 'a;
}

pub struct TCell<'a, T, SHAPE> {
    cell: Ref<'a, Vec<T>>,
    offset: usize,
    _phantom: PhantomData<SHAPE>,
}

pub struct TCellMut<'a, T, SHAPE> {
    cell: RefMut<'a, Vec<T>>,
    offset: usize,
    _phantom: PhantomData<SHAPE>,
}

impl<'a, T, const D0: usize> Indexable for TCell<'a, T, V<D0>> {
    type IndexType = usize;
}

impl<'b, T: Float, const RW: bool, const D0: usize> TReader<T, V<D0>>
    for Tensor<'b, RW, T, V<D0>, RefStore<'b, T>>
{
    type Reader<'a> = TCell<'a, T, V<D0>> where Self: 'a,  T: 'a;
    fn reader(&self) -> Self::Reader<'_> {
        let (offset, cell) = self.store;
        let cell = cell.borrow();
        TCell {
            cell,
            offset,
            _phantom: PhantomData,
        }
    }
}

impl<'b, T: Float, const RW: bool, const D0: usize> TWriter<T, V<D0>>
    for Tensor<'b, RW, T, V<D0>, RefStore<'b, T>>
{
    type Writer<'a> = TCellMut<'a, T, V<D0>> where Self: 'a,  T: 'a;
    fn writer(&mut self) -> Self::Writer<'_> {
        let (offset, cell) = self.store;
        let cell = cell.borrow_mut();
        TCellMut {
            cell,
            offset,
            _phantom: PhantomData,
        }
    }
}

impl<'b, T: Float, const RW: bool, const D0: usize, const D1: usize> TReader<T, M<D0, D1>>
    for Tensor<'b, RW, T, M<D0, D1>, RefStore<'b, T>>
{
    type Reader<'a> = TCell<'a, T, M<D0, D1>> where Self: 'a,  T: 'a;
    fn reader(&self) -> Self::Reader<'_> {
        let (offset, cell) = self.store;
        let cell = cell.borrow();
        TCell {
            cell,
            offset,
            _phantom: PhantomData,
        }
    }
}

impl<'b, T: Float, const RW: bool, const D0: usize, const D1: usize> TWriter<T, M<D0, D1>>
    for Tensor<'b, RW, T, M<D0, D1>, RefStore<'b, T>>
{
    type Writer<'a> = TCellMut<'a, T, M<D0, D1>> where Self: 'a,  T: 'a;
    fn writer(&mut self) -> Self::Writer<'_> {
        let (offset, cell) = self.store;
        let cell = cell.borrow_mut();
        TCellMut {
            cell,
            offset,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Copy, const D0: usize> TGet<usize> for TCell<'a, T, V<D0>> {
    type Output = T;
    fn get(&self, index: usize) -> Self::Output {
        let x = self.cell.deref();
        #[cfg(not(debug_assertions))]
        unsafe {
            *x.get_unchecked(index)
        }
        #[cfg(debug_assertions)]
        x[index]
    }
}

impl<'a, T: Copy, const D0: usize> TGet<usize> for TCellMut<'a, T, V<D0>> {
    type Output = T;
    fn get(&self, index: usize) -> Self::Output {
        let x = self.cell.deref();
        #[cfg(not(debug_assertions))]
        unsafe {
            *x.get_unchecked(index)
        }
        #[cfg(debug_assertions)]
        x[index]
    }
}

impl<'a, T: Copy, const D0: usize> TSet<usize> for TCellMut<'a, T, V<D0>> {
    fn set(&mut self, index: usize, val: T) {
        let x = self.cell.deref_mut();
        #[cfg(not(debug_assertions))]
        unsafe {
            *x.get_unchecked_mut(index) = val
        }
        #[cfg(debug_assertions)]
        {
            x[index] = val
        }
    }
}

impl<'a, T: Copy, const D0: usize, const D1: usize> TGet<(usize, usize)>
    for TCell<'a, T, M<D0, D1>>
{
    type Output = T;
    fn get(&self, (i, j): (usize, usize)) -> Self::Output {
        let x = self.cell.deref();
        let index = self.offset + i * D0 + j;
        #[cfg(not(debug_assertions))]
        unsafe {
            *x.get_unchecked(index)
        }
        #[cfg(debug_assertions)]
        x[index]
    }
}

impl<'a, T: Copy, const D0: usize, const D1: usize> TGet<(usize, usize)>
    for TCellMut<'a, T, M<D0, D1>>
{
    type Output = T;
    fn get(&self, (i, j): (usize, usize)) -> Self::Output {
        let x = self.cell.deref();
        let index = self.offset + i * D0 + j;
        #[cfg(not(debug_assertions))]
        unsafe {
            *x.get_unchecked(index)
        }
        #[cfg(debug_assertions)]
        x[index]
    }
}

impl<'a, T: Copy, const D0: usize, const D1: usize> TSet<(usize, usize)>
    for TCellMut<'a, T, M<D0, D1>>
{
    fn set(&mut self, (i, j): (usize, usize), val: T) {
        let x = self.cell.deref_mut();
        let index = self.offset + i * D0 + j;
        #[cfg(not(debug_assertions))]
        unsafe {
            *x.get_unchecked_mut(index) = val
        }
        #[cfg(debug_assertions)]
        {
            x[index] = val
        }
    }
}

impl<'a, T: Float, const D0: usize, const D1: usize> RowableMut<T, D0, D1, RefStore<'a, T>>
    for Tensor<'a, true, T, MATRIX<D0, D1>, RefStore<'a, T>>
{
    type RowStoreType = RefStore<'a, T>;
    type RowTensorType<'c> = Tensor<'a, true, T, V<D0>, RefStore<'a, T>> where Self: 'c;
    fn row<'b>(&'b mut self, i: usize) -> Self::RowTensorType<'b> {
        let (offset, cell) = self.store;
        assert!(offset == 0);
        Tensor {
            store: (offset + i * D0, cell),
        }
    }
}

impl<'a, T: Float, const D0: usize, const D1: usize> Rowable<T, D0, D1, RefStore<'a, T>>
    for Tensor<'a, false, T, MATRIX<D0, D1>, RefStore<'a, T>>
{
    type RowStoreType = RefStore<'a, T>;
    type RowTensorType<'c> = Tensor<'a, false, T, V<D0>, RefStore<'a, T>> where Self: 'c;
    fn row<'b>(&'b self, i: usize) -> Self::RowTensorType<'b> {
        let (offset, cell) = self.store;
        assert!(offset == 0);
        Tensor {
            store: (offset + i * D0, cell),
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
    type ReaderTypeA<'a> = TSlice<'a, T, SHAPE> where T: 'a;
    type WriterTypeA<'a> = () where T: 'a;
}

impl<T, SHAPE: IsTensor> TensorTypes<T, SHAPE> for SubStore<T, true> {
    type StoreType<'a> = &'a mut [T] where T: 'a;
    type ReaderType<'a> = &'a [T] where T: 'a;
    type WriterType<'a> = &'a mut [T] where T: 'a;
    type ReaderTypeA<'a> = TSlice<'a, T, SHAPE> where T: 'a;
    type WriterTypeA<'a> = TSliceMut<'a, T, SHAPE> where T: 'a;
}

impl<SHAPE: IsTensor> TensorTypes<f32, SHAPE> for SubStore<f16> {
    type StoreType<'a> = &'a [f16];
    type ReaderType<'a> = &'a [f16];
    type WriterType<'a> = &'a mut [f16];
    type ReaderTypeA<'a> = TSlice<'a, f16, SHAPE>;
    type WriterTypeA<'a> = TSliceMut<'a, f16, SHAPE>;
}

impl<'a, T, SHAPE: IsTensor + Indexable> TReader<T, SHAPE>
    for Tensor<'a, false, T, SHAPE, SubStore<T, false>>
where
    for<'c> TSlice<'c, T, SHAPE>: TGet<<SHAPE as Indexable>::IndexType, Output = T>,
{
    type Reader<'b> = TSlice<'b, T, SHAPE> where Self: 'b;
    fn reader(&self) -> Self::Reader<'_> {
        TSlice {
            slice: self.store,
            _phantom0: PhantomData,
            _phantom1: PhantomData,
        }
    }
}

impl<'a, T, SHAPE: IsTensor + Indexable> TReader<T, SHAPE>
    for Tensor<'a, true, T, SHAPE, SubStore<T, true>>
where
    for<'c> TSlice<'c, T, SHAPE>: TGet<<SHAPE as Indexable>::IndexType, Output = T>,
{
    type Reader<'b> = TSlice<'b, T, SHAPE> where Self: 'b;
    fn reader(&self) -> Self::Reader<'_> {
        TSlice {
            slice: self.store,
            _phantom0: PhantomData,
            _phantom1: PhantomData,
        }
    }
}

impl<'a, T: Copy, const D0: usize> TWriter<T, V<D0>>
    for Tensor<'a, true, T, V<D0>, SubStore<T, true>>
where
    for<'c> TSlice<'c, T, V<D0>>: TGet<<V<D0> as Indexable>::IndexType, Output = T>,
    for<'c> TSliceMut<'c, T, V<D0>>: TSet<<V<D0> as Indexable>::IndexType, Output = T>,
{
    type Writer<'b> = TSliceMut<'b, T, V<D0>> where Self: 'b;
    fn writer(&mut self) -> Self::Writer<'_> {
        TSliceMut {
            slice: self.store,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Copy, const D0: usize, const D1: usize> TWriter<T, M<D0, D1>>
    for Tensor<'a, true, T, M<D0, D1>, SubStore<T, true>>
where
    for<'c> TSlice<'c, T, M<D0, D1>>: TGet<<M<D0, D1> as Indexable>::IndexType, Output = T>,
    for<'c> TSliceMut<'c, T, M<D0, D1>>: TSet<<M<D0, D1> as Indexable>::IndexType, Output = T>,
{
    type Writer<'b> = TSliceMut<'b, T, M<D0, D1>> where Self: 'b;
    fn writer(&mut self) -> Self::Writer<'_> {
        TSliceMut {
            slice: self.store,
            _phantom: PhantomData,
        }
    }
}

impl<'a, SHAPE: IsTensor + Indexable> TReader<f32, SHAPE, f16>
    for Tensor<'a, false, f32, SHAPE, SubStore<f16, false>>
where
    SubStore<f16>: TensorTypes<f16, SHAPE>,
    for<'c> TSlice<'c, f32, SHAPE, f16>: TGet<<SHAPE as Indexable>::IndexType, Output = f32>,
{
    type Reader<'b> = TSlice<'b, f32, SHAPE, f16> where Self: 'b;
    fn reader(&self) -> Self::Reader<'_> {
        TSlice {
            slice: self.store,
            _phantom0: PhantomData,
            _phantom1: PhantomData,
        }
    }
}

impl<'a, SHAPE: IsTensor + Indexable> TReader<f32, SHAPE>
    for Tensor<'a, false, f32, SHAPE, SubStore<f16, false>>
where
    SubStore<f16>: TensorTypes<f16, SHAPE>,
    for<'c> TVec<f32, SHAPE>: TGet<<SHAPE as Indexable>::IndexType, Output = f32>,
{
    type Reader<'b> = TVec<f32, SHAPE> where Self: 'b;
    fn reader(&self) -> Self::Reader<'_> {
        TVec {
            vec: self.store.iter().map(|&x| x.into()).collect(),
            _phantom: PhantomData,
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
    type ReaderTypeA<'a> = TSlice<'a, T, SHAPE> where T: 'a;
    type WriterTypeA<'a> = TSliceMut<'a, T, SHAPE> where T: 'a;
}

impl<'a, T, SHAPE: IsTensor + Indexable> TReader<T, SHAPE>
    for Tensor<'a, false, T, SHAPE, VecStore<T>>
where
    for<'c> TSlice<'c, T, SHAPE>: TGet<<SHAPE as Indexable>::IndexType, Output = T>,
{
    type Reader<'b> = TSlice<'b, T, SHAPE> where Self: 'b;
    fn reader(&self) -> Self::Reader<'_> {
        TSlice {
            slice: self.store.as_slice(),
            _phantom0: PhantomData,
            _phantom1: PhantomData,
        }
    }
}

impl<SHAPE: IsTensor> TensorTypes<f32, SHAPE> for VecStore<f16> {
    type StoreType<'a> = Vec<f16>;
    type ReaderType<'a> = &'a [f16];
    type WriterType<'a> = ();
    type ReaderTypeA<'a> = TSlice<'a, f16, SHAPE>;
    type WriterTypeA<'a> = ();
}

impl<'a, SHAPE: IsTensor + Indexable> TReader<f32, SHAPE>
    for Tensor<'a, false, f32, SHAPE, VecStore<f16>>
where
    TVec<f32, SHAPE>: TGet<<SHAPE as Indexable>::IndexType, Output = f32>,
{
    type Reader<'b> = TVec<f32, SHAPE> where Self: 'b;
    fn reader(&self) -> Self::Reader<'_> {
        TVec {
            vec: self.store.iter().map(|x| (*x).into()).collect(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Copy, const D0: usize, const D1: usize> Rowable<T, D0, D1, VecStore<T>>
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

impl<'a, const D0: usize, const D1: usize> Rowable<f32, D0, D1, VecStore<f16>>
    for Tensor<'a, false, f32, MATRIX<D0, D1>, VecStore<f16>>
{
    type RowStoreType = SubStore<f16>;
    type RowTensorType<'b> = Tensor<'b, false, f32, VECTOR<D0>, SubStore<f16>> where Self: 'b;

    fn row<'b>(&'b self, i: usize) -> Self::RowTensorType<'b> {
        debug_assert!(i < D1);
        Tensor {
            store: &self.store.as_slice()[i * D0..(i + 1) * D0],
        }
    }
}
