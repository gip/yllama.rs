use crate::*;
use std::marker::PhantomData;

// Shape //////////////////////////////////////////////////////////////////////
pub trait Tensor {
    fn n_elem() -> usize;
    fn dim() -> usize;
}

trait Matrix: Tensor {}
trait Vector: Tensor {}

pub struct VECTOR<const D0: usize> {}
pub type V<const D0: usize> = VECTOR<D0>;

impl<const D0: usize> Tensor for VECTOR<D0> {
    fn n_elem() -> usize {
        D0
    }
    fn dim() -> usize { 1 }
}
impl<const D0: usize> Vector for VECTOR<D0> {}

pub struct MATRIX<const D0: usize, const D1: usize> {}
pub type M<const D0: usize, const D1: usize> = MATRIX<D0, D1>;
impl<const D0:usize, const D1: usize> Tensor for MATRIX<D0, D1> {
    fn n_elem() -> usize {
        D0 * D1
    }
    fn dim() -> usize { 2 }
}
impl<const D0: usize, const D1: usize> Vector for MATRIX<D0, D1> {}


// Tensor /////////////////////////////////////////////////////////////////////
// pub trait D<const DIM: usize> {
//     fn shape(&self) -> [usize; DIM];
// }

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

pub trait TReader<T, SHAPE: Tensor> {
    type Reader<'b>: TRead<T, SHAPE>
    where
        Self: 'b;
    fn reader<'a>(&'a self) -> Self::Reader<'a>;
}

pub trait TRead<T, SHAPE: Tensor> {
    fn reading(&self) -> &[T];
}

pub trait TWriter<T: Copy, SHAPE: Tensor> {
    type Writer<'b>: TWrite<T, SHAPE>
    where
        Self: 'b;
    fn writer<'a>(&'a mut self) -> Self::Writer<'a>;
}

pub trait TWrite<T: Copy, SHAPE: Tensor> {
    fn writing(&mut self) -> &mut [T];
}

// TensorMut //////////////////////////////////////////////////////////////////
pub struct TensorMut<'a, T, SHAPE> {
    phantom: PhantomData<SHAPE>,
    pub vec: Option<Vec<T>>,
    pub slice: &'a mut [T],
}

pub type VectorMut<'a, T, SHAPE: Vector> = TensorMut<'a, T, SHAPE>;
pub type Tensor2Mut<'a, T, SHAPE: Matrix> = TensorMut<'a, T, SHAPE>;

// impl<'a, T, const DIM: usize> D<DIM> for TensorMut<'a, T, DIM> {
//     fn shape(&self) -> [usize; DIM] {
//         self.shape
//     }
// }

impl<'a, T, SHAPE: Tensor> TReader<T, SHAPE> for TensorMut<'a, T, SHAPE> {
    type Reader<'b> = &'b [T] where Self: 'b;
    fn reader(&self) -> &[T] {
        self.slice
    }
}

impl<'a, T: Copy, SHAPE: Tensor> TWriter<T, SHAPE> for TensorMut<'a, T, SHAPE> {
    type Writer<'b> = &'b mut [T] where Self: 'b;
    fn writer(&mut self) -> &mut [T] {
        self.slice
    }
}

impl<'a, T, SHAPE: Tensor> TRead<T, SHAPE> for TensorMut<'a, T, SHAPE> {
    fn reading(&self) -> &[T] {
        self.slice
    }
}

impl<'a, T: Copy, SHAPE: Tensor> TWrite<T, SHAPE> for TensorMut<'a, T, SHAPE> {
    fn writing(&mut self) -> &mut [T] {
        self.slice
    }
}

impl<'a, T: Float, SHAPE: Vector> VectorMut<'a, T, SHAPE> {
    pub fn new(i: usize) -> Self {
        let mut vec = vec![T::zero(); i];
        let slice: &'a mut [T] = unsafe { std::mem::transmute(vec.as_mut_slice()) };
        VectorMut {
            phantom: PhantomData,
            vec: Some(vec),
            slice,
        }
    }

    // pub fn new_from(y: &'a mut Vector<'a, T, SubStore<T>>) -> Self {
    //     let ([size], y_slice) = y.reader();
    //     let mut vec = vec![T::zero(); size];
    //     let slice: &'a mut [T] = unsafe { std::mem::transmute(vec.as_mut_slice()) };
    //     for i in 0..size {
    //         vec[i] = y_slice[i];
    //     }
    //     VectorMut {
    //         shape: [size],
    //         vec: Some(vec),
    //         slice,
    //     }
    // }
}

// impl<'a, T: Float> Tensor2Mut<'a, T> {
//     pub fn new(i: usize, j: usize) -> Self {
//         let mut vec = vec![T::zero(); i * j];
//         let slice: &'a mut [T] = unsafe { std::mem::transmute(vec.as_mut_slice()) };
//         Tensor2Mut {
//             shape: [i, j],
//             vec: Some(vec),
//             slice,
//         }
//     }
// }

impl<'a, T, SHAPE: Vector> Index<usize> for TensorMut<'a, T, SHAPE> {
    type Output = T;
    fn index<'b>(&'b self, index: usize) -> &'b Self::Output {
        let slice = self.reading();
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

impl<'a, T: Copy, SHAPE: Vector> IndexMut<usize> for TensorMut<'a, T, SHAPE> {
    // Output defined in Index trait, T
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let slice = self.writing();
        #[cfg(debug_assertions)]
        unsafe {
            slice.get_unchecked_mut(index)
        }
        #[cfg(not(debug_assertions))]
        {
            slice.get_mut(index).unwrap() // TODO: check this, why do we need to unwrap?
        }
    }
}

impl<'a, T, const D0: usize, const D1: usize> Index<(usize, usize)> for TensorMut<'a, T, MATRIX<D0, D1>> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        debug_assert!(i < D1 && j < D0);
        unsafe { &self.slice.get_unchecked(i * D0 + j) }
    }
}

impl<'a, T, const D0: usize, const D1: usize> IndexMut<(usize, usize)> for TensorMut<'a, T, MATRIX<D0, D1>> {
    // Output defined in Index trait, T
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        debug_assert!(i < D1 && j < D0);
        unsafe { self.slice.get_unchecked_mut(i * D0 + j) }
        //&mut self.slice[i * d0 + j]
    }
}

// TODO -> abstract with Rowable trait?
impl<'a, T: Copy, const D0: usize, const D1: usize> TensorMut<'a, T, MATRIX<D0, D1>> {
    pub fn row(&mut self, i: usize) -> VectorMut<T, VECTOR<D1>> {
        let slice = self.writing();
        debug_assert!(i < D0);
        VectorMut {
            phantom: PhantomData,
            vec: None,
            slice: &mut slice[i * D1..(i + 1) * D1],
        }
    }
}

// Tensor /////////////////////////////////////////////////////////////////////
pub struct TensorImm<'a, T: 'a, SHAPE: Tensor, U: TensorTypes<T, SHAPE>> {
    pub store: U::StoreType<'a>,
}

pub type VectorImm<'a, T, SHAPE: Vector, U> = TensorImm<'a, T, SHAPE, U>;
pub type Tensor2Imm<'a, T, SHAPE: Matrix, U> = TensorImm<'a, T, SHAPE, U>;

// MmapStore //////////////////////////////////////////////////////////////////
pub struct MmapStore<T, U, const CP: bool = true> {
    phantom_t: PhantomData<T>,
    phantom_u: PhantomData<U>,
}

// Strategy: data on mmap, no copy
impl<T, SHAPE: Tensor> TensorTypes<T, SHAPE> for MmapStore<T, T> {
    type StoreType<'a> = (Rc<Mmap>, &'a [T]) where T: 'a;
    type ReaderType<'a> = &'a [T] where T: 'a;
    type WriterType<'a> = &'a mut [T] where T: 'a;
}

impl<'a, T, SHAPE: Tensor> TReader<T, SHAPE> for TensorImm<'a, T, SHAPE, MmapStore<T, T>> {
    type Reader<'b> = &'b [T] where Self: 'b;
    fn reader<'c>(&'c self) -> Self::Reader<'c> {
        self.store.1
    }
}

// Strategy: f16 data on mmap, full data conversion + on-demand copy to a Vec<f32>
impl<SHAPE: Tensor> TensorTypes<f32, SHAPE> for MmapStore<f32, f16, true> {
    type StoreType<'a> = (Rc<Mmap>, &'a [f16]);
    type ReaderType<'a> = Vec<f32>;
    type WriterType<'a> = &'a mut [f16];
}

impl<'a, SHAPE: Tensor> TReader<f32, SHAPE> for TensorImm<'a, f32, SHAPE, MmapStore<f32, f16, true>> {
    type Reader<'b> = Vec<f32> where Self: 'b;
    fn reader<'c>(&'c self) -> Self::Reader<'c> {
        self.store.1.iter().map(|&value| f32::from(value)).collect()
    }
}

// Strategy: f16 data on mmap, on the fly data conversion
impl<SHAPE: Tensor> TensorTypes<f32, SHAPE> for MmapStore<f32, f16, false> {
    type StoreType<'a> = (Rc<Mmap>, &'a [f16]);
    type ReaderType<'a> = &'a [f16];
    type WriterType<'a> = &'a mut [f16];
}

// impl<'a, const DIM: usize> TReader<f32, DIM> for Tensor<'a, f32, DIM, MmapStore<f32, f16, false>>
// {
//     type Reader<'b> = ([usize; DIM], Vec<f32>) where Self: 'b;
//     fn reader<'c>(&'c self) -> Self::Reader<'c> {
//         let slice = self.store.1.iter().map(|&value| f32::from(value)).collect();
//         (self.shape, slice)
//     }
// }

impl<'a, const D0: usize, const D1: usize> Rowable<'a, f32, VECTOR<D0>, MmapStore<f32, f32>> for TensorImm<'a, f32, MATRIX<D0, D1>, MmapStore<f32, f32>> {
    fn row(&self, i: usize) -> impl TReader<f32, VECTOR<D0>> {
        debug_assert!(i < D1);
        let v: VectorImm<'a, f32, VECTOR<D0>, MmapStore<f32, f32>> = VectorImm {
            store: (self.store.0.clone(), &self.store.1[i * D0..(i + 1) * D0]),
        };
        v
    }
}

impl<'a, const D0: usize, const D1: usize> Rowable<'a, f32, VECTOR<D0>, MmapStore<f32, f16>> for TensorImm<'a, f32, MATRIX<D0, D1>, MmapStore<f32, f16>> {
    fn row(&self, i: usize) -> impl TReader<f32, VECTOR<D0>> {
        debug_assert!(i < D1);
        let v: VectorImm<'a, f32, VECTOR<D0>, MmapStore<f32, f16>> = VectorImm {
            store: (self.store.0.clone(), &self.store.1[i * D0..(i + 1) * D0]),
        };
        v
    }
}

// SubStore ///////////////////////////////////////////////////////////////////
pub struct SubStore<U> {
    phantom: PhantomData<U>,
}

impl<T, SHAPE: Tensor> TensorTypes<T, SHAPE> for SubStore<T> {
    type StoreType<'a> = &'a [T] where T: 'a;
    type ReaderType<'a> = &'a [T] where T: 'a;
    type WriterType<'a> = &'a mut [T] where T: 'a;
}

impl<'a, T, SHAPE: Tensor + 'a> TReader<T, SHAPE> for TensorImm<'a, T, SHAPE, SubStore<T>> {
    type Reader<'b> = &'b [T] where Self: 'b;
    fn reader(&self) -> Self::Reader<'a> {
        &self.store
    }
}

impl<'a, T, SHAPE: Vector + 'a> Index<usize> for TensorImm<'a, T, SHAPE, SubStore<T>> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        let reader = self.reader();
        let slice = <&[T] as tensor::TRead<T, SHAPE>>::reading(&reader);
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
pub struct VecStore<U, F> {
    phantom: PhantomData<U>,
    phantom_f: PhantomData<F>,
}

impl<T, U, SHAPE: Tensor> TensorTypes<T, SHAPE> for VecStore<T, U> {
    type StoreType<'a> = Vec<T> where T: 'a;
    type ReaderType<'a> = &'a [T] where T: 'a;
    type WriterType<'a> = &'a mut [T] where T: 'a;
}

impl<'a, T, SHAPE: Tensor, F> TReader<T, SHAPE> for TensorImm<'a, T, SHAPE, VecStore<T, F>> {
    type Reader<'b> = &'b [T] where Self: 'b;
    fn reader<'c>(&'c self) -> Self::Reader<'c> {
        &self.store.as_slice()
    }
}

// TRead //////////////////////////////////////////////////////////////////////
impl<'a, 'b, T, SHAPE: Tensor> TRead<T, SHAPE> for &'b [T] {
    fn reading(&self) -> &[T] {
        self
    }
}

impl<'a, 'b, T, SHAPE: Tensor> TRead<T, SHAPE> for Vec<T> {
    fn reading(&self) -> &[T] {
        self.as_slice()
    }
}

// TWrite /////////////////////////////////////////////////////////////////////
impl<'a: 'b, 'b, T: Copy, SHAPE: Tensor> TWrite<T, SHAPE> for &'b mut [T] {
    fn writing(&mut self) ->  &mut [T] {
        self
    }
}

//
pub trait Rowable<'a, T, SHAPE: Vector, U>
where
    U: TensorTypes<T, SHAPE>,
{
    fn row(&self, i: usize) -> impl TReader<T, SHAPE>;
}
