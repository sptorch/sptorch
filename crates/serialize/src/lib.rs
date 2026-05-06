use sptorch_core_tensor::Tensor;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

pub mod safetensors;

const MAGIC: u32 = 0x5350_5443; // "SPTC"
const VERSION: u32 = 1;

/// Save model parameters to a binary checkpoint file.
/// Format: [magic:u32][version:u32][num_params:u32]
///         for each param: [ndim:u32][shape...:u32][data...:f32]
pub fn save_checkpoint<P: AsRef<Path>>(path: P, params: &[Tensor]) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    w.write_all(&MAGIC.to_le_bytes())?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(params.len() as u32).to_le_bytes())?;

    for p in params {
        let shape = p.shape();
        let data = p.contiguous_data();

        w.write_all(&(shape.len() as u32).to_le_bytes())?;
        for &dim in &shape {
            w.write_all(&(dim as u32).to_le_bytes())?;
        }
        for &val in &data {
            w.write_all(&val.to_le_bytes())?;
        }
    }

    w.flush()?;
    Ok(())
}

/// Load checkpoint data into existing parameter tensors.
/// The number and shapes of params must match the checkpoint.
pub fn load_checkpoint<P: AsRef<Path>>(path: P, params: &[Tensor]) -> io::Result<()> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut buf4 = [0u8; 4];

    r.read_exact(&mut buf4)?;
    let magic = u32::from_le_bytes(buf4);
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid checkpoint magic"));
    }

    r.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported checkpoint version {}", version),
        ));
    }

    r.read_exact(&mut buf4)?;
    let num_params = u32::from_le_bytes(buf4) as usize;
    if num_params != params.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "param count mismatch: checkpoint has {}, model has {}",
                num_params,
                params.len()
            ),
        ));
    }

    for (i, p) in params.iter().enumerate() {
        r.read_exact(&mut buf4)?;
        let ndim = u32::from_le_bytes(buf4) as usize;

        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            r.read_exact(&mut buf4)?;
            shape.push(u32::from_le_bytes(buf4) as usize);
        }

        let expected_shape = p.shape();
        if shape != expected_shape {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "shape mismatch at param[{}]: checkpoint {:?}, model {:?}",
                    i, shape, expected_shape
                ),
            ));
        }

        let numel: usize = shape.iter().product();
        let mut data = vec![0.0f32; numel];
        for val in data.iter_mut() {
            r.read_exact(&mut buf4)?;
            *val = f32::from_le_bytes(buf4);
        }

        // Write data into the parameter's storage
        let inner = p.0.read().unwrap();
        let mut storage = inner.storage.write().unwrap();
        storage.as_cpu_slice_mut().copy_from_slice(&data);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_save_load_roundtrip() {
        let p1 = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let p2 = Tensor::with_grad(vec![5.0, 6.0], vec![2], true);
        let params = vec![p1.clone(), p2.clone()];

        let path = "test_checkpoint.sptc";
        save_checkpoint(path, &params).unwrap();

        // Create new params with different values
        let q1 = Tensor::with_grad(vec![0.0; 4], vec![2, 2], true);
        let q2 = Tensor::with_grad(vec![0.0; 2], vec![2], true);
        let new_params = vec![q1.clone(), q2.clone()];

        load_checkpoint(path, &new_params).unwrap();

        assert_eq!(q1.data(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(q2.data(), vec![5.0, 6.0]);

        fs::remove_file(path).unwrap();
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_load_shape_mismatch() {
        let p1 = Tensor::new(vec![1.0, 2.0], vec![2]);
        let path = "test_checkpoint_mismatch.sptc";
        save_checkpoint(path, &[p1]).unwrap();

        let q1 = Tensor::new(vec![0.0; 4], vec![4]);
        let result = load_checkpoint(path, &[q1]);
        assert!(result.is_err());

        fs::remove_file(path).unwrap();
    }
}
