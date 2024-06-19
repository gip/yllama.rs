#[path = "gguf_file.rs"]
pub mod gguf_file;
pub use gguf_file::*;

use bytes::{BufMut, BytesMut};
use std::borrow::Borrow;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

type E = Box<dyn std::error::Error>;

pub fn read_gguf_file<'a>(fname: PathBuf, read_buffer_size: usize) -> Result<GGUFFile<()>, E> {
    let mut buffer = BytesMut::with_capacity(read_buffer_size);
    let mut reader = BufReader::with_capacity(read_buffer_size, File::open(fname)?);
    loop {
        let read: &[u8] = reader.fill_buf()?;
        if read.is_empty() {
            return Err("Failed to read gguf file".into());
        }
        let content_length = read.len();
        buffer.put(read);
        reader.consume(content_length);
        match GGUFFile::read(buffer.borrow()) {
            Ok(Some(file)) => {
                return Ok(file);
            }
            Ok(None) => {
                // skip
            }
            Err(e) => {
                return Err(e.into());
            }
        }
        buffer.reserve(read_buffer_size);
    }
}
