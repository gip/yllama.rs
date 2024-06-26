use crate::*;

// Tensor /////////////////////////////////////////////////////////////////////
pub trait D<const DIM: usize> {
    fn shape(&self) -> [usize; DIM];
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

pub trait TWrite<T: Copy, const DIM: usize> {
    fn writing(&mut self) -> ([usize; DIM], &mut [T]);
}

// TensorMut //////////////////////////////////////////////////////////////////
pub struct TensorMut<'a, T, const DIM: usize> {
    pub shape: [usize; DIM],
    pub vec: Option<Vec<T>>,
    pub slice: &'a mut [T],
}

pub type VectorMut<'a, T> = TensorMut<'a, T, 1>;
pub type Tensor2Mut<'a, T> = TensorMut<'a, T, 2>;

impl<'a, T, const DIM: usize> D<DIM> for TensorMut<'a, T, DIM> {
    fn shape(&self) -> [usize; DIM] {
        self.shape
    }
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

impl<'a, T: Float> VectorMut<'a, T> {
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

impl<'a, T: Float> Tensor2Mut<'a, T> {
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

impl<'a, T> Index<usize> for TensorMut<'a, T, 1> {
    type Output = T;
    fn index<'b>(&'b self, index: usize) -> &'b Self::Output {
        let (_, slice) = self.reading();
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

impl<'a, T: Copy> IndexMut<usize> for TensorMut<'a, T, 1> {
    // Output defined in Index trait, T
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let (_, slice) = self.writing();
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

// TODO -> abstract with Rowable trait?
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

// Tensor /////////////////////////////////////////////////////////////////////
pub struct Tensor<'a, T: 'a, const DIM: usize, U: TensorTypes<T, DIM>> {
    pub shape: [usize; DIM],
    pub store: U::StoreType<'a>,
}

pub type Vector<'a, T, U> = Tensor<'a, T, 1, U>;
pub type Tensor2<'a, T, U> = Tensor<'a, T, 2, U>;

impl<'a, T: From<U>, U, const DIM: usize> D<DIM> for Tensor<'a, T, DIM, U>
where
    U: TensorTypes<T, DIM>,
{
    fn shape(&self) -> [usize; DIM] {
        self.shape
    }
}

// MmapStore //////////////////////////////////////////////////////////////////
pub struct MmapStore<T, U, const CP: bool = true> {
    phantom_t: PhantomData<T>,
    phantom_u: PhantomData<U>,
}

// Strategy: data on mmap, no copy
impl<T, const DIM: usize> TensorTypes<T, DIM> for MmapStore<T, T> {
    type Shape = [usize; DIM];
    type StoreType<'a> = (Rc<Mmap>, &'a [T]) where T: 'a;
    type ReaderType<'a> = (Self::Shape, &'a [T]) where T: 'a;
    type WriterType<'a> = (Self::Shape, &'a mut [T]) where T: 'a;
}

impl<'a, T, const DIM: usize> TReader<T, DIM> for Tensor<'a, T, DIM, MmapStore<T, T>> {
    type Reader<'b> = ([usize; DIM], &'b [T]) where Self: 'b;
    fn reader<'c>(&'c self) -> Self::Reader<'c> {
        (self.shape, &self.store.1)
    }
}

// Strategy: f16 data on mmap, full data conversion + on-demand copy to a Vec<f32>
impl<const DIM: usize> TensorTypes<f32, DIM> for MmapStore<f32, f16, true> {
    type Shape = [usize; DIM];
    type StoreType<'a> = (Rc<Mmap>, &'a [f16]);
    type ReaderType<'a> = (Self::Shape, Vec<f32>);
    type WriterType<'a> = (Self::Shape, &'a mut [f16]);
}

impl<'a, const DIM: usize> TReader<f32, DIM> for Tensor<'a, f32, DIM, MmapStore<f32, f16, true>> {
    type Reader<'b> = ([usize; DIM], Vec<f32>) where Self: 'b;
    fn reader<'c>(&'c self) -> Self::Reader<'c> {
        let slice = self.store.1.iter().map(|&value| f32::from(value)).collect();
        (self.shape, slice)
    }
}

// Strategy: f16 data on mmap, on the fly data conversion
impl<const DIM: usize> TensorTypes<f32, DIM> for MmapStore<f32, f16, false> {
    type Shape = [usize; DIM];
    type StoreType<'a> = (Rc<Mmap>, &'a [f16]);
    type ReaderType<'a> = (Self::Shape, &'a [f16]);
    type WriterType<'a> = (Self::Shape, &'a mut [f16]);
}

// impl<'a, const DIM: usize> TReader<f32, DIM> for Tensor<'a, f32, DIM, MmapStore<f32, f16, false>>
// {
//     type Reader<'b> = ([usize; DIM], Vec<f32>) where Self: 'b;
//     fn reader<'c>(&'c self) -> Self::Reader<'c> {
//         let slice = self.store.1.iter().map(|&value| f32::from(value)).collect();
//         (self.shape, slice)
//     }
// }

impl<'a> Rowable<'a, f32, MmapStore<f32, f32>> for Tensor<'a, f32, 2, MmapStore<f32, f32>> {
    fn row(&self, i: usize) -> impl TReader<f32, 1> {
        let ([d0, d1], _) = self.reader().reading();
        debug_assert!(i < d1);
        let v: Vector<'a, f32, MmapStore<f32, f32>> = Vector {
            shape: [d0],
            store: (self.store.0.clone(), &self.store.1[i * d0..(i + 1) * d0]),
        };
        v
    }
}

impl<'a> Rowable<'a, f32, MmapStore<f32, f16>> for Tensor<'a, f32, 2, MmapStore<f32, f16>> {
    fn row(&self, i: usize) -> impl TReader<f32, 1> {
        let ([d0, d1], _) = self.reader().reading();
        debug_assert!(i < d1);
        let v: Vector<'a, f32, MmapStore<f32, f16>> = Vector {
            shape: [d0],
            store: (self.store.0.clone(), &self.store.1[i * d0..(i + 1) * d0]),
        };
        v
    }
}

// SubStore ///////////////////////////////////////////////////////////////////
pub struct SubStore<U> {
    phantom: PhantomData<U>,
}

impl<T, const DIM: usize> TensorTypes<T, DIM> for SubStore<T> {
    type Shape = [usize; DIM];
    type StoreType<'a> = &'a [T] where T: 'a;
    type ReaderType<'a> = (Self::Shape, &'a [T]) where T: 'a;
    type WriterType<'a> = (Self::Shape, &'a mut [T]) where T: 'a;
}

impl<'a, T, const DIM: usize> TReader<T, DIM> for Tensor<'a, T, DIM, SubStore<T>> {
    type Reader<'b> = ([usize; DIM], &'b [T]) where Self: 'b;
    fn reader(&self) -> Self::Reader<'a> {
        (self.shape, &self.store)
    }
}

impl<'a, T> Index<usize> for Tensor<'a, T, 1, SubStore<T>> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        let reader = self.reader();
        let slice = reader.reading().1;
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

impl<T, U, const DIM: usize> TensorTypes<T, DIM> for VecStore<T, U> {
    type Shape = [usize; DIM];
    type StoreType<'a> = Vec<T> where T: 'a;
    type ReaderType<'a> = (Self::Shape, &'a [T]) where T: 'a;
    type WriterType<'a> = (Self::Shape, &'a mut [T]) where T: 'a;
}

impl<'a, T, const DIM: usize, F> TReader<T, DIM> for Tensor<'a, T, DIM, VecStore<T, F>> {
    type Reader<'b> = ([usize; DIM], &'b [T]) where Self: 'b;
    fn reader<'c>(&'c self) -> Self::Reader<'c> {
        (self.shape, &self.store.as_slice())
    }
}

// TRead //////////////////////////////////////////////////////////////////////
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

// TWrite /////////////////////////////////////////////////////////////////////
impl<'a: 'b, 'b, T: Copy, const DIM: usize> TWrite<T, DIM> for ([usize; DIM], &'b mut [T]) {
    fn writing(&mut self) -> ([usize; DIM], &mut [T]) {
        (self.0, self.1)
    }
}

//
pub trait Rowable<'a, T, U>
where
    U: TensorTypes<T, 1>,
{
    fn row(&self, i: usize) -> impl TReader<T, 1>;
}
