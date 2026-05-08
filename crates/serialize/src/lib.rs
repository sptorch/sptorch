use sptorch_core_tensor::Tensor;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

pub mod safetensors;

const MAGIC: u32 = 0x5350_5443; // "SPTC"
const VERSION: u32 = 1;

/// Save model parameters to a binary checkpoint file.
/// Format: `[magic:u32][version:u32][num_params:u32]`
///         for each param: `[ndim:u32][shape...:u32][data...:f32]`
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
/// 按参数名导出的稳定权重快照条目。
#[derive(Debug, Clone)]
pub struct StateDictEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: sptorch_core_tensor::DType,
    pub data: Vec<f32>,
}

/// 导出类似 PyTorch `state_dict` 的参数快照。
///
/// 参数名是稳定键，调用方应确保同一模型中名称唯一。这里返回的是内存
/// 结构而不是文件格式，方便后续直接写入 JSON、checkpoint 或 safetensors。
pub fn export_state_dict(params: &[(&str, Tensor)]) -> Vec<StateDictEntry> {
    params
        .iter()
        .map(|(name, tensor)| StateDictEntry {
            name: (*name).to_string(),
            shape: tensor.shape(),
            dtype: tensor.dtype(),
            data: tensor.contiguous_data(),
        })
        .collect()
}

/// 按名称回写参数快照到模型张量。
///
/// 这个接口会严格检查名称、shape 和 dtype。只要其中任一项不匹配，就会
/// 返回错误，避免把错误权重静默灌入模型。
pub fn load_state_dict(params: &[(&str, Tensor)], entries: &[StateDictEntry]) -> io::Result<()> {
    let by_name: std::collections::HashMap<&str, &StateDictEntry> =
        entries.iter().map(|entry| (entry.name.as_str(), entry)).collect();

    for (name, param) in params {
        let entry = by_name.get(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("state_dict missing parameter '{name}'"),
            )
        })?;
        if entry.shape != param.shape() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "shape mismatch for '{name}': state {:?}, model {:?}",
                    entry.shape,
                    param.shape()
                ),
            ));
        }
        if entry.dtype != param.dtype() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "dtype mismatch for '{name}': state {:?}, model {:?}",
                    entry.dtype,
                    param.dtype()
                ),
            ));
        }
        let inner = param.0.read().unwrap();
        let mut storage = inner.storage.write().unwrap();
        storage.as_cpu_slice_mut().copy_from_slice(&entry.data);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_state_dict_roundtrip_by_name() {
        let w = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let b = Tensor::with_grad(vec![5.0, 6.0], vec![2], true);
        let state = export_state_dict(&[("linear.weight", w.clone()), ("linear.bias", b.clone())]);

        let new_w = Tensor::with_grad(vec![0.0; 4], vec![2, 2], true);
        let new_b = Tensor::with_grad(vec![0.0; 2], vec![2], true);
        load_state_dict(
            &[("linear.weight", new_w.clone()), ("linear.bias", new_b.clone())],
            &state,
        )
        .unwrap();

        assert_eq!(new_w.data(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(new_b.data(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_state_dict_rejects_shape_mismatch() {
        let w = Tensor::new(vec![1.0, 2.0], vec![2]);
        let state = export_state_dict(&[("w", w)]);
        let target = Tensor::new(vec![0.0; 4], vec![4]);
        let err = load_state_dict(&[("w", target)], &state).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

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
