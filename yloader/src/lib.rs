pub use gguf::gguf_file::{
    header_find_f32, header_find_string, header_find_string_array, header_find_usize, GGUFFile,
    GGUFTensor,
};
use half::f16;
use memmap2::{Mmap, MmapOptions};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::fs::File;
use std::mem::size_of;
use std::rc::Rc;

mod gguf;
pub use gguf::GGMLType;
use gguf::{read_gguf_file, GGUFHeader};
use ymath::{MmapStore, Tensor, TensorTypes};

#[derive(Debug)]
pub struct ModelFile<E = ()> {
    #[allow(dead_code)]
    _path: String,
    pub mmap: Rc<Mmap>,
    pub header: GGUFHeader,
    pub tensors: BTreeMap<String, GGUFTensor<E>>,
    pub tensor_data_offset: u64,
}

pub trait Tensorify<'a, T, const DIM: usize, U, MF>
where
    U: TensorTypes<T, DIM>,
{
    fn to_tensor(&self, mf: MF) -> Result<Tensor<'a, T, DIM, U>, Box<dyn std::error::Error>>;
}

impl<'a, const DIM: usize> Tensorify<'a, f32, DIM, MmapStore<f32>, &ModelFile> for GGUFTensor<()> {
    fn to_tensor(
        &self,
        model: &ModelFile,
    ) -> Result<Tensor<'a, f32, DIM, MmapStore<f32>>, Box<dyn std::error::Error>> {
        let d = self.dimensions.len();
        if DIM != d {
            return Err(anyhow::anyhow!("wrong dimension for tensor '{}'", &self.name).into());
        };
        let mut shape: [usize; DIM] = [0; DIM];
        for i in 0..d {
            shape[i] = self.dimensions[i] as usize
        }
        let n_elem: usize = self.dimensions.iter().fold(1, |a, b| a * b) as usize;
        let mmap = &model.mmap;
        let tensor_data_offset = model.tensor_data_offset;
        let base_ptr = mmap.as_ptr();
        let data = unsafe { base_ptr.add((tensor_data_offset + self.relative_offset) as usize) };
        let slice = match self.tensor_type {
            GGMLType::F32 => unsafe {
                std::slice::from_raw_parts(data as *const f32, size_of::<f32>() * n_elem)
            },
            _ => return Err(anyhow::anyhow!("wrong type for tensor '{}'", &self.name).into()),
        };
        Ok(Tensor {
            shape,
            store: (Rc::clone(mmap), slice),
        })
    }
}

impl<'a, const DIM: usize> Tensorify<'a, f32, DIM, MmapStore<f16>, &ModelFile> for GGUFTensor<()> {
    fn to_tensor(
        &self,
        model: &ModelFile,
    ) -> Result<Tensor<'a, f32, DIM, MmapStore<f16>>, Box<dyn std::error::Error>> {
        let d = self.dimensions.len();
        if DIM != d {
            return Err(anyhow::anyhow!("wrong dimension for tensor '{}'", &self.name).into());
        };
        let mut shape: [usize; DIM] = [0; DIM];
        for i in 0..d {
            shape[i] = self.dimensions[i] as usize
        }
        let n_elem: usize = self.dimensions.iter().fold(1, |a, b| a * b) as usize;
        let mmap = &model.mmap;
        let tensor_data_offset = model.tensor_data_offset;
        let base_ptr = mmap.as_ptr();
        let data = unsafe { base_ptr.add((tensor_data_offset + self.relative_offset) as usize) };
        let slice = match self.tensor_type {
            GGMLType::F16 => unsafe {
                std::slice::from_raw_parts(data as *const f16, size_of::<f16>() * n_elem)
            },
            _ => return Err(anyhow::anyhow!("wrong type for tensor '{}'", &self.name).into()),
        };
        Ok(Tensor {
            shape,
            store: (Rc::clone(mmap), slice),
        })
    }
}

// Load a GGUF file
pub fn load_fast<'a>(
    path: &str,
) -> Result<(String, String, GGUFFile<()>), Box<dyn std::error::Error>> {
    let gguf = read_gguf_file(path.into(), 1_000_000)?;
    let arch = header_find_string(&gguf.header, "general.architecture")?;
    let name = header_find_string(&gguf.header, "general.name")?;
    Ok((arch.to_string(), name.to_string(), gguf))
}

pub fn load_build<'a>(
    path: &str,
    gguf: GGUFFile<()>,
) -> Result<ModelFile, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().populate().map(&file)? };
    let mmap_rc = Rc::new(mmap);
    let tensor_data_offset = gguf.tensor_data_offset;

    Ok(ModelFile {
        mmap: mmap_rc,
        _path: path.to_string(),
        header: gguf.header,
        tensors: gguf.tensors,
        tensor_data_offset,
    })
}
