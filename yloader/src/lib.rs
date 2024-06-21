pub use gguf::gguf_file::{
    header_find_f32, header_find_string, header_find_string_array, header_find_usize, GGUFFile,
    GGUFTensor,
};
use memmap2::{Mmap, MmapOptions};
use std::mem::size_of;
use std::thread;
use std::{collections::BTreeMap, fs::File};

mod gguf;
use gguf::{read_gguf_file, GGMLType};
use ymath::{
    dequantize_row_q4_k, dequantize_row_q6_k, BlockQ4K, BlockQ6K, MemLayout, Tensor2, Tensorify2,
    Vector, Vectorify,
};

#[derive(Debug)]
pub struct ModelFile<T> {
    #[allow(dead_code)]
    mmap: Mmap, // This is never read but needs to be kept in scope
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
        }
    }
}

fn tensor_build_layout<'a, T>(
    base_ptr: *const u8,
    tensor_data_offset: u64,
    tensor: GGUFTensor<T>,
) -> MemLayout<'a, f32> {
    let data = unsafe { base_ptr.add((tensor_data_offset + tensor.relative_offset) as usize) };
    let n_elem: usize = tensor.dimensions.iter().fold(1, |a, b| a * b) as usize;
    let slice = match tensor.tensor_type {
        GGMLType::F32 => MemLayout::MmapLayout {
            slice: unsafe {
                std::slice::from_raw_parts(data as *const f32, size_of::<f32>() * n_elem)
            },
        },
        GGMLType::Q4K => {
            let mut vec: Vec<f32> = vec![0.0; n_elem];
            let blocks: &[BlockQ4K] = unsafe {
                std::slice::from_raw_parts(data as *const BlockQ4K, size_of::<BlockQ4K>() * n_elem)
            };
            dequantize_row_q4_k(blocks, &mut vec, n_elem as usize);
            MemLayout::VecLayout { vec }
        }
        GGMLType::Q6K => {
            let mut vec: Vec<f32> = vec![0.0; n_elem];
            let blocks: &[BlockQ6K] = unsafe {
                std::slice::from_raw_parts(data as *const BlockQ6K, size_of::<BlockQ6K>() * n_elem)
            };
            dequantize_row_q6_k(blocks, &mut vec, n_elem as usize);
            MemLayout::VecLayout { vec }
        }
        _ => unimplemented!(),
    };
    slice
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
) -> Result<ModelFile<GGUFFile<MemLayout<'a, f32>>>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().populate().map(&file)? };

    let mut tensors = BTreeMap::new();

    // Load tensor (theaded version available)
    if true {
        tensors = gguf
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
                        ext: tensor_build_layout(mmap.as_ptr(), gguf.tensor_data_offset, g),
                    },
                )
            })
            .collect();
    } else {
        thread::scope(|s| {
            let handles: Vec<_> = gguf
                .tensors
                .into_iter()
                .map(|(k, g)| {
                    s.spawn(|| {
                        (
                            k,
                            GGUFTensor {
                                name: g.name.clone(),
                                dimensions: g.dimensions.clone(),
                                tensor_type: g.tensor_type,
                                relative_offset: g.relative_offset,
                                ext: tensor_build_layout(mmap.as_ptr(), gguf.tensor_data_offset, g),
                            },
                        )
                    })
                })
                .collect();

            tensors = handles.into_iter().map(|h| h.join().unwrap()).collect();
        });
    }

    let model = GGUFFile {
        header: gguf.header,
        tensors,
        tensor_data_offset: gguf.tensor_data_offset,
    };

    Ok(ModelFile {
        mmap,
        model,
        _path: path.to_string(),
    })
}
