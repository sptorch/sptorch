use serde::Deserialize;
use sptorch_core_tensor::Tensor;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// A loaded safetensors file: name -> (Tensor, shape).
pub struct SafeTensorsFile {
    pub tensors: HashMap<String, Tensor>,
}

impl SafeTensorsFile {
    /// Load a .safetensors file from disk.
    ///
    /// Format: [header_size: u64 LE][header: JSON][tensor data bytes...]
    /// Header JSON maps tensor names to {dtype, shape, data_offsets: [start, end]}.
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() < 8 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        let header_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if 8 + header_size > bytes.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "header size exceeds file"));
        }

        let header_json = &bytes[8..8 + header_size];
        let header: HashMap<String, serde_json::Value> = serde_json::from_slice(header_json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("invalid header JSON: {}", e)))?;

        let data_start = 8 + header_size;
        let mut tensors = HashMap::new();

        for (name, value) in &header {
            if name == "__metadata__" {
                continue;
            }

            let info: TensorInfo = serde_json::from_value(value.clone()).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bad tensor info for '{}': {}", name, e),
                )
            })?;

            let tensor_bytes = &bytes[data_start + info.data_offsets[0]..data_start + info.data_offsets[1]];
            let data = convert_to_f32(&info.dtype, tensor_bytes)?;

            let tensor = Tensor::new(data, info.shape);
            tensors.insert(name.clone(), tensor);
        }

        Ok(SafeTensorsFile { tensors })
    }
    /// `get`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }
    /// `names`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
    /// `load_into`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn load_into(&self, params: &[Tensor], mapping: &[(usize, &str)]) -> io::Result<()> {
        for &(idx, name) in mapping {
            let src = self.tensors.get(name).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("tensor '{}' not found in safetensors", name),
                )
            })?;

            let src_data = src.data();
            let param = &params[idx];
            let param_shape = param.shape();
            let src_shape = src.shape();

            if param_shape != src_shape {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "shape mismatch for '{}': safetensors {:?}, model {:?}",
                        name, src_shape, param_shape
                    ),
                ));
            }

            let inner = param.0.read().unwrap();
            let mut storage = inner.storage.write().unwrap();
            storage.as_cpu_slice_mut().copy_from_slice(&src_data);
        }
        Ok(())
    }
}

// 按 safetensors 的 dtype 将原始字节流转换为 f32 向量。
fn convert_to_f32(dtype: &str, bytes: &[u8]) -> io::Result<Vec<f32>> {
    match dtype {
        "F32" => {
            if !bytes.len().is_multiple_of(4) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "F32 data not aligned"));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        "F16" => {
            if !bytes.len().is_multiple_of(2) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "F16 data not aligned"));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes(c.try_into().unwrap());
                    f16_to_f32(bits)
                })
                .collect())
        }
        "BF16" => {
            if !bytes.len().is_multiple_of(2) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "BF16 data not aligned"));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes(c.try_into().unwrap());
                    bf16_to_f32(bits)
                })
                .collect())
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported dtype: {}", dtype),
        )),
    }
}

// 将 IEEE 754 half (f16) 位模式手动扩展为 f32。
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // subnormal
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exp == 31 {
        let f32_bits = (sign << 31) | (0xFF << 23) | (mant << 13);
        return f32::from_bits(f32_bits);
    }

    let f32_exp = exp + 127 - 15;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

// 将 bfloat16 通过高 16 位对齐方式转换为 f32。
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    // 验证 f16 常见值转换结果。
    #[test]
    fn test_f16_to_f32_basic() {
        // f16 for 1.0: sign=0, exp=15, mant=0 -> bits = 0x3C00
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 1e-6);

        // f16 for 0.0
        let val = f16_to_f32(0x0000);
        assert_eq!(val, 0.0);

        // f16 for -1.0: 0xBC00
        let val = f16_to_f32(0xBC00);
        assert!((val - (-1.0)).abs() < 1e-6);
    }

    // 验证 bf16 常见值转换结果。
    #[test]
    fn test_bf16_to_f32_basic() {
        // bf16 for 1.0: 0x3F80 (upper 16 bits of f32 1.0 = 0x3F800000)
        let val = bf16_to_f32(0x3F80);
        assert!((val - 1.0).abs() < 1e-6);

        let val = bf16_to_f32(0x0000);
        assert_eq!(val, 0.0);
    }

    // 验证 F32 字节序列可被正确解析。
    #[test]
    fn test_convert_f32() {
        let data: Vec<u8> = vec![1.0f32, 2.0, 3.0].iter().flat_map(|f| f.to_le_bytes()).collect();
        let result = convert_to_f32("F32", &data).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    // 构造最小 safetensors 文件，验证加载与张量形状/数据。
    #[test]
    fn test_load_safetensors_synthetic() {
        // Build a minimal safetensors file in memory
        let tensor_data: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let header = serde_json::json!({
            "weight": {
                "dtype": "F32",
                "shape": [2, 3],
                "data_offsets": [0, 24]
            }
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let header_size = header_bytes.len() as u64;

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_size.to_le_bytes());
        file_bytes.extend_from_slice(&header_bytes);
        file_bytes.extend_from_slice(&tensor_data);

        let path = "test_synthetic.safetensors";
        std::fs::write(path, &file_bytes).unwrap();

        let st = SafeTensorsFile::load(path).unwrap();
        assert!(st.names().contains(&"weight"));

        let t = st.get("weight").unwrap();
        assert_eq!(t.shape(), vec![2, 3]);
        assert_eq!(t.data(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        std::fs::remove_file(path).unwrap();
    }

    // 验证按名称映射将 safetensors 权重写回模型参数。
    #[test]
    fn test_load_into_params() {
        // Build safetensors file
        let tensor_data: Vec<u8> = vec![10.0f32, 20.0, 30.0, 40.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let header = serde_json::json!({
            "layer.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16]
            }
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let header_size = header_bytes.len() as u64;

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_size.to_le_bytes());
        file_bytes.extend_from_slice(&header_bytes);
        file_bytes.extend_from_slice(&tensor_data);

        let path = "test_load_into.safetensors";
        std::fs::write(path, &file_bytes).unwrap();

        let st = SafeTensorsFile::load(path).unwrap();

        let param = Tensor::new(vec![0.0; 4], vec![2, 2]);
        st.load_into(&[param.clone()], &[(0, "layer.weight")]).unwrap();
        assert_eq!(param.data(), vec![10.0, 20.0, 30.0, 40.0]);

        std::fs::remove_file(path).unwrap();
    }
}
