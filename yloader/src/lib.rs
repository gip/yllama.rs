pub use gguf::gguf_file::{
    header_find_f32, header_find_string, header_find_string_array, header_find_usize, GGUFFile,
    GGUFTensor,
};
use memmap2::{Mmap, MmapOptions};
use std::fmt::{Debug, Formatter};
use std::mem::size_of;
use std::rc::Rc;
use std::fs::File;

mod gguf;
use gguf::{read_gguf_file, GGMLType};
use ymath::{Tensor2, Tensorify2, Vector, Vectorify};

// Memory layout
#[derive(Clone)]
pub enum MemLayout<'a, T> {
    MmapLayout { slice: &'a [T], mmap: Rc<Mmap> },
    VecLayout { vec: Vec<T> },
}

impl<'a, T> MemLayout<'a, T> {
    pub fn to_slice(&self) -> &[T] {
        match self {
            MemLayout::MmapLayout { slice, mmap: _ } => slice,
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

#[derive(Debug)]
pub struct ModelFile<T> {
    #[allow(dead_code)]
    _path: String,
    pub model: T,
}

impl<'a, T> Tensorify2<'a, T> for GGUFTensor<MemLayout<'a, T>>
where
    T: Clone,
{
    fn to_tensor2(&self) -> Tensor2<T> {
        assert!(self.dimensions.len() == 2);
        Tensor2 {
            shape: [self.dimensions[0] as usize, self.dimensions[1] as usize],
            vec: None,
            slice: self.ext.to_slice(),
            ext: (),
        }
    }
}

impl<'a, T> Vectorify<'a, T> for GGUFTensor<MemLayout<'a, T>>
where
    T: Clone,
{
    fn to_vector(&self) -> Vector<T> {
        assert!(self.dimensions.len() == 1);
        Vector {
            shape: [self.dimensions[0] as usize],
            vec: None,
            slice: self.ext.to_slice(),
            ext: (),
        }
    }
}

// pub trait TensorBuildLayout<'a, T, U> {
//     fn build_mmap_layout(
//         mmap: Mmap,
//         tensor_data_offset: u64,
//         tensor: GGUFTensor<T>) -> MemLayout<'a, U>;
// }

fn build_mmap_layout<'a, T, U>(
    mmap: Rc<Mmap>,
    tensor_data_offset: u64,
    tensor: GGUFTensor<T>,
) -> MemLayout<'a, U> {
    let base_ptr = mmap.as_ptr();
    let data = unsafe { base_ptr.add((tensor_data_offset + tensor.relative_offset) as usize) };
    let n_elem: usize = tensor.dimensions.iter().fold(1, |a, b| a * b) as usize;
    let slice = match tensor.tensor_type {
        GGMLType::F32 => MemLayout::MmapLayout {
            slice: unsafe {
                std::slice::from_raw_parts(data as *const U, size_of::<f32>() * n_elem)
            },
            mmap: mmap.into(),
        },
        _ => todo!(),
    };
    slice
}

// trait Val {
//     fn val() -> GGMLType;
// }

// impl Val for f32 {
//     fn val() -> GGMLType {
//         GGMLType::F32
//     }
// }
// impl Val for f16 {
//     fn val() -> GGMLType {
//         GGMLType::F16
//     }
// }

// fn _build_mmap_layout<'a, T, U>(
//     mmap: Rc<Mmap>,
//     tensor_data_offset: u64,
//     tensor: GGUFTensor<T>,
// ) -> Result<MemLayout<'a, U>, anyhow::Error>
// where
//     U: Val,
// {
//     let base_ptr = mmap.as_ptr();
//     let data = unsafe { base_ptr.add((tensor_data_offset + tensor.relative_offset) as usize) };
//     let n_elem: usize = tensor.dimensions.iter().fold(1, |a, b| a * b) as usize;
//     let val = U::val();
//     if val == tensor.tensor_type {
//         Ok(MemLayout::MmapLayout {
//             slice: unsafe {
//                 std::slice::from_raw_parts(data as *const U, size_of::<f32>() * n_elem)
//             },
//             mmap: mmap.into(),
//         })
//     } else {
//         Err(anyhow::anyhow!("Wrong tensor type"))
//     }
// }

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
) -> Result<ModelFile<GGUFFile<MemLayout<'a, f32>>>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().populate().map(&file)? };
    let mmap_rc = Rc::new(mmap);

    let tensors = gguf
        .tensors
        .into_iter()
        .map(|(k, g)| {
            (
                k,
                GGUFTensor {
                    name: g.name.clone(),
                    dimensions: g.dimensions.clone(),
                    tensor_type: g.tensor_type,
                    relative_offset: g.relative_offset,
                    ext: build_mmap_layout(mmap_rc.clone(), gguf.tensor_data_offset, g),
                },
            )
        })
        .collect();

    let model = GGUFFile {
        header: gguf.header,
        tensors,
        tensor_data_offset: gguf.tensor_data_offset,
    };

    Ok(ModelFile {
        model,
        _path: path.to_string(),
    })
}
