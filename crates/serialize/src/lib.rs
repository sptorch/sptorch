use serde::{Deserialize, Serialize};
use sptorch_core_tensor::{DType, Tensor};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

pub mod safetensors;

const MAGIC: u32 = 0x5350_5443; // "SPTC"
const VERSION: u32 = 1;
/// JSON state_dict 的稳定 schema 名称。
///
/// 这个常量会写入文件头部，也会被 versioning 的 manifest 引用。这样训练产物
/// 不只保存权重本身，还能明确说明自己遵循哪一版命名参数格式。
pub const STATE_DICT_SCHEMA: &str = "sptorch.state_dict.v1";
/// 当前 JSON state_dict 文件格式版本。
pub const STATE_DICT_FORMAT_VERSION: u32 = 1;

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
    pub dtype: DType,
    pub data: Vec<f32>,
}

/// 一个模型级别的命名参数快照。
#[derive(Debug, Clone)]
pub struct NamedStateDict {
    pub entries: Vec<StateDictEntry>,
}

impl NamedStateDict {
    /// 用命名参数对构造快照。
    pub fn from_named_params(params: &[(&str, Tensor)]) -> Self {
        Self {
            entries: export_state_dict(params),
        }
    }

    /// 把快照回写到命名参数集合。
    pub fn load_into(&self, params: &[(&str, Tensor)]) -> io::Result<()> {
        load_state_dict(params, &self.entries)
    }
}

/// 可落盘的 state_dict 条目。
///
/// `core-tensor::DType` 不直接依赖 serde，因此文件格式在 serialize crate 内部
/// 使用字符串保存 dtype，避免把序列化依赖反向压到底层张量库。
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableStateDictEntry {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    data: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableStateDictFile {
    schema: String,
    format_version: u32,
    entries: Vec<SerializableStateDictEntry>,
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

/// 保存命名参数快照到 JSON state_dict 文件。
pub fn save_named_state_dict<P: AsRef<Path>>(path: P, params: &[(&str, Tensor)]) -> io::Result<()> {
    save_state_dict(path, params)
}

/// 从 JSON state_dict 文件读取命名参数快照，并回写到模型。
pub fn load_named_state_dict<P: AsRef<Path>>(path: P, params: &[(&str, Tensor)]) -> io::Result<NamedStateDict> {
    let entries = load_state_dict_file(path, params)?;
    Ok(NamedStateDict { entries })
}

/// 按名称回写参数快照到模型张量。
///
/// 这个接口会严格检查名称、shape 和 dtype。只要其中任一项不匹配，就会
/// 返回错误，避免把错误权重静默灌入模型。
pub fn load_state_dict(params: &[(&str, Tensor)], entries: &[StateDictEntry]) -> io::Result<()> {
    if entries.len() != params.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "state_dict parameter count mismatch: state has {}, model has {}",
                entries.len(),
                params.len()
            ),
        ));
    }

    let mut by_name = std::collections::HashMap::new();
    for entry in entries {
        if by_name.insert(entry.name.as_str(), entry).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("state_dict contains duplicate parameter '{}'", entry.name),
            ));
        }
    }

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
        if entry.data.len() != param.numel() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "data length mismatch for '{name}': state {}, model {}",
                    entry.data.len(),
                    param.numel()
                ),
            ));
        }
        let inner = param.0.read().unwrap();
        let mut storage = inner.storage.write().unwrap();
        storage.as_cpu_slice_mut().copy_from_slice(&entry.data);
    }

    Ok(())
}

/// 把命名参数保存为 JSON state_dict 文件。
///
/// 这个格式优先服务调试、训练闭环测试和轻量产品集成；生产级大权重可以继续走
/// 二进制 checkpoint 或 safetensors。文件中包含 schema 与版本号，便于未来做
/// 向后兼容迁移。
pub fn save_state_dict<P: AsRef<Path>>(path: P, params: &[(&str, Tensor)]) -> io::Result<()> {
    let file = SerializableStateDictFile {
        schema: STATE_DICT_SCHEMA.into(),
        format_version: STATE_DICT_FORMAT_VERSION,
        entries: export_state_dict(params)
            .into_iter()
            .map(|entry| SerializableStateDictEntry {
                name: entry.name,
                shape: entry.shape,
                dtype: dtype_to_string(entry.dtype).into(),
                data: entry.data,
            })
            .collect(),
    };
    let f = File::create(path)?;
    serde_json::to_writer_pretty(BufWriter::new(f), &file)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

/// 从 JSON state_dict 文件加载命名参数。
pub fn load_state_dict_file<P: AsRef<Path>>(path: P, params: &[(&str, Tensor)]) -> io::Result<Vec<StateDictEntry>> {
    let f = File::open(path)?;
    let file: SerializableStateDictFile =
        serde_json::from_reader(BufReader::new(f)).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    if file.schema != STATE_DICT_SCHEMA {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported state_dict schema {}", file.schema),
        ));
    }
    if file.format_version != STATE_DICT_FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported state_dict version {}", file.format_version),
        ));
    }

    let entries: Vec<StateDictEntry> = file
        .entries
        .into_iter()
        .map(|entry| {
            Ok(StateDictEntry {
                name: entry.name,
                shape: entry.shape,
                dtype: dtype_from_string(&entry.dtype)?,
                data: entry.data,
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    load_state_dict(params, &entries)?;
    Ok(entries)
}

fn dtype_to_string(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "F32",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
    }
}

fn dtype_from_string(dtype: &str) -> io::Result<DType> {
    match dtype {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported dtype '{other}'"),
        )),
    }
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
    fn test_state_dict_rejects_extra_parameter() {
        let w = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0], vec![1]);
        let state = export_state_dict(&[("w", w), ("b", b)]);
        let target = Tensor::new(vec![0.0; 2], vec![2]);
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

    #[test]
    fn test_named_state_dict_roundtrip_file() {
        let w = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let b = Tensor::with_grad(vec![5.0, 6.0], vec![2], true);
        let path = "test_named_state_dict.json";

        save_named_state_dict(path, &[("linear.weight", w.clone()), ("linear.bias", b.clone())]).unwrap();

        let new_w = Tensor::with_grad(vec![0.0; 4], vec![2, 2], true);
        let new_b = Tensor::with_grad(vec![0.0; 2], vec![2], true);
        let loaded = load_named_state_dict(
            path,
            &[("linear.weight", new_w.clone()), ("linear.bias", new_b.clone())],
        )
        .unwrap();
        assert_eq!(loaded.entries.len(), 2);
        assert_eq!(new_w.data(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(new_b.data(), vec![5.0, 6.0]);

        fs::remove_file(path).unwrap();
    }
}
