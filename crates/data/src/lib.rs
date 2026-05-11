//! 轻量数据管道：tokenizer、文本序列数据集与批加载器。
//!
//! 这个 crate 刻意保持依赖很薄，用来支撑框架级训练示例和产品仓的最小
//! 数据输入路径。它不负责复杂语料清洗，也不持有 Tensor；输出保持为 token id
//! 序列，方便上层自行决定是否进入 CPU、CUDA 或自研硬件后端。

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;

// ============ Tokenizer Trait ============

/// 文本与 token id 之间的最小转换契约。
///
/// 这里不强制特殊 token、padding 或 unknown token 语义，因为不同产品会有不同
/// 词表策略。实现者需要保证 `decode(encode(text))` 在可表示字符集合内尽量稳定。
pub trait Tokenizer {
    /// 将输入文本转换为 token id。不可表示的内容如何处理由具体 tokenizer 决定。
    fn encode(&self, text: &str) -> Vec<usize>;
    /// 将 token id 序列还原为字符串；调用方必须保证 id 位于词表范围内。
    fn decode(&self, ids: &[usize]) -> String;
    /// 返回当前词表大小，用于初始化 embedding 或输出投影层。
    fn vocab_size(&self) -> usize;
}

// ============ CharTokenizer ============

/// 字符级 tokenizer，适合小语料、教学样例和框架 smoke test。
///
/// 词表由训练文本中出现过的 Unicode 字符组成，并按排序后的稳定顺序分配 id。
/// 未出现在词表中的字符会在 `encode` 时被跳过，这一点对生产产品通常过于简单，
/// 但非常适合框架层验证训练闭环。
pub struct CharTokenizer {
    pub vocab: Vec<char>,
    char_to_id: HashMap<char, usize>,
}

impl CharTokenizer {
    /// 从一段文本构建字符词表。
    ///
    /// 重复字符会被去重；词表顺序稳定，因此同一段文本在不同进程中会得到一致 id。
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let char_to_id: HashMap<char, usize> = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        CharTokenizer {
            vocab: chars,
            char_to_id,
        }
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().filter_map(|c| self.char_to_id.get(&c).copied()).collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| self.vocab[id]).collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ============ BPE Tokenizer ============

/// 极简 BPE tokenizer。
///
/// 它实现的是“按相邻 token 频次贪心合并”的核心思想，目标是给 SPTorch 的训练
/// 基线提供可解释的子词压缩能力，而不是完整复刻 HuggingFace tokenizer 生态。
pub struct BpeTokenizer {
    vocab: Vec<String>, // id -> token string
    token_to_id: HashMap<String, usize>,
    merges: Vec<(usize, usize)>, // ordered merge rules: (id_a, id_b) -> new_id
}

impl BpeTokenizer {
    /// 在单段语料上训练一个小型 BPE 词表。
    ///
    /// `target_vocab_size` 是上限而不是保证值；当语料没有可继续合并的相邻 token
    /// 时会提前结束。合并规则按训练时顺序保存，编码阶段也按同一顺序应用。
    pub fn train(text: &str, target_vocab_size: usize) -> Self {
        // 先以字符作为初始 token，保证任意训练语料都能被表示。
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut vocab: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        let mut token_to_id: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

        let mut token_ids: Vec<usize> = text
            .chars()
            .map(|c| *token_to_id.get(&c.to_string()).unwrap())
            .collect();

        let mut merges = Vec::new();

        while vocab.len() < target_vocab_size {
            let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
            for w in token_ids.windows(2) {
                *pair_counts.entry((w[0], w[1])).or_insert(0) += 1;
            }

            if pair_counts.is_empty() {
                break;
            }

            let &best_pair = pair_counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(pair, _)| pair)
                .unwrap();

            let (a, b) = best_pair;
            let new_token = format!("{}{}", vocab[a], vocab[b]);
            let new_id = vocab.len();
            vocab.push(new_token.clone());
            token_to_id.insert(new_token, new_id);
            merges.push(best_pair);

            // 合并必须在同一次扫描中跳过右侧 token，避免 `aaa` 中同一字符被重复消费。
            let mut new_ids = Vec::with_capacity(token_ids.len());
            let mut i = 0;
            while i < token_ids.len() {
                if i + 1 < token_ids.len() && token_ids[i] == a && token_ids[i + 1] == b {
                    new_ids.push(new_id);
                    i += 2;
                } else {
                    new_ids.push(token_ids[i]);
                    i += 1;
                }
            }
            token_ids = new_ids;
        }

        BpeTokenizer {
            vocab,
            token_to_id,
            merges,
        }
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids: Vec<usize> = text
            .chars()
            .filter_map(|c| self.token_to_id.get(&c.to_string()).copied())
            .collect();

        for &(a, b) in &self.merges {
            let new_id = self.token_to_id.get(&format!("{}{}", self.vocab[a], self.vocab[b]));
            let new_id = match new_id {
                Some(&id) => id,
                None => continue,
            };
            let mut new_ids = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
                    new_ids.push(new_id);
                    i += 2;
                } else {
                    new_ids.push(ids[i]);
                    i += 1;
                }
            }
            ids = new_ids;
        }
        ids
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| self.vocab[id].as_str()).collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ============ Dataset Trait ============

/// 可被 DataLoader 批量读取的样本集合。
///
/// 样本返回 `(input, target)` 两条 token 序列，适合语言模型 next-token 训练。
/// 这里没有抽象成关联类型，是为了让当前框架训练入口保持非常直接。
pub trait Dataset {
    /// 样本数量。
    fn len(&self) -> usize;
    /// 判断数据集是否为空；默认实现依赖 `len`。
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// 返回指定样本。调用方必须保证 `index < len()`。
    fn get(&self, index: usize) -> (Vec<usize>, Vec<usize>);
}

// ============ TextDataset ============

/// next-token 文本数据集。
///
/// 对长度为 `seq_len` 的输入窗口，target 是向右平移一位的窗口。若 token 总数不足
/// `seq_len + 1`，数据集长度为 0。
pub struct TextDataset {
    tokens: Vec<usize>,
    seq_len: usize,
}

impl TextDataset {
    /// 创建固定窗口长度的数据集。
    pub fn new(tokens: Vec<usize>, seq_len: usize) -> Self {
        TextDataset { tokens, seq_len }
    }
}

impl Dataset for TextDataset {
    fn len(&self) -> usize {
        if self.tokens.len() <= self.seq_len {
            0
        } else {
            self.tokens.len() - self.seq_len
        }
    }

    fn get(&self, index: usize) -> (Vec<usize>, Vec<usize>) {
        let input = self.tokens[index..index + self.seq_len].to_vec();
        let target = self.tokens[index + 1..index + self.seq_len + 1].to_vec();
        (input, target)
    }
}

// ============ DataLoader ============

/// 简单批加载器。
///
/// 它在构造时生成索引表，并在 `reset` 时按需重新 shuffle。批次允许最后一批小于
/// `batch_size`，这对小语料训练和 benchmark 更友好。
pub struct DataLoader<'a, D: Dataset> {
    dataset: &'a D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    pos: usize,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    /// 创建新的批加载器。
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            indices,
            pos: 0,
        }
    }

    /// 将读取位置回到开头；若启用 shuffle，会重新打乱样本顺序。
    pub fn reset(&mut self) {
        self.pos = 0;
        if self.shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
    }

    /// 读取下一批 `(inputs, targets)`。
    ///
    /// 返回 `None` 表示当前 epoch 已耗尽。每个 batch 内部保留样本级 Vec，方便上层
    /// 决定是否拼成二维 Tensor 或保留变长序列。
    #[allow(clippy::type_complexity)]
    pub fn next_batch(&mut self) -> Option<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        if self.pos >= self.indices.len() {
            return None;
        }

        let end = (self.pos + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.pos..end];
        self.pos = end;

        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for &idx in batch_indices {
            let (inp, tgt) = self.dataset.get(idx);
            inputs.push(inp);
            targets.push(tgt);
        }

        Some((inputs, targets))
    }

    /// 返回当前索引表会被切成多少个 batch。
    pub fn num_batches(&self) -> usize {
        self.indices.len().div_ceil(self.batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 字符 tokenizer 的基础验收：词表去重、编码长度和 roundtrip 都必须稳定。
    #[test]
    fn test_char_tokenizer() {
        let tok = CharTokenizer::from_text("hello");
        assert_eq!(tok.vocab_size(), 4); // e, h, l, o
        let ids = tok.encode("hello");
        assert_eq!(ids.len(), 5);
        assert_eq!(tok.decode(&ids), "hello");
    }

    // 较长文本 roundtrip 用来防止词表排序或 id 映射被误改。
    #[test]
    fn test_char_tokenizer_roundtrip() {
        let text = "the quick brown fox";
        let tok = CharTokenizer::from_text(text);
        let ids = tok.encode(text);
        assert_eq!(tok.decode(&ids), text);
    }

    // TextDataset 的核心不变量是 target 永远比 input 右移一位。
    #[test]
    fn test_text_dataset() {
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let ds = TextDataset::new(tokens, 4);
        assert_eq!(ds.len(), 4); // 8 - 4 = 4 valid positions
        let (inp, tgt) = ds.get(0);
        assert_eq!(inp, vec![0, 1, 2, 3]);
        assert_eq!(tgt, vec![1, 2, 3, 4]);
    }

    // 非 shuffle 路径应保持顺序读取，并在耗尽后返回 None。
    #[test]
    fn test_dataloader_no_shuffle() {
        let tokens = vec![0, 1, 2, 3, 4, 5];
        let ds = TextDataset::new(tokens, 2);
        // len = 6 - 2 = 4 samples
        let mut dl = DataLoader::new(&ds, 2, false);
        let batch1 = dl.next_batch().unwrap();
        assert_eq!(batch1.0.len(), 2);
        let batch2 = dl.next_batch().unwrap();
        assert_eq!(batch2.0.len(), 2);
        assert!(dl.next_batch().is_none());
    }

    // reset 是训练循环复用 DataLoader 的关键路径。
    #[test]
    fn test_dataloader_reset() {
        let tokens = vec![0, 1, 2, 3, 4];
        let ds = TextDataset::new(tokens, 2);
        let mut dl = DataLoader::new(&ds, 10, false);
        let _ = dl.next_batch();
        assert!(dl.next_batch().is_none());
        dl.reset();
        assert!(dl.next_batch().is_some());
    }

    // 批次数使用向上取整，确保最后一个不满 batch 不会被丢弃。
    #[test]
    fn test_dataloader_num_batches() {
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let ds = TextDataset::new(tokens, 3);
        let dl = DataLoader::new(&ds, 3, false);
        assert_eq!(dl.num_batches(), 3);
    }

    // --- BPE Tokenizer tests ---

    #[test]
    fn test_bpe_train_basic() {
        let text = "aaabdaaabac";
        let tok = BpeTokenizer::train(text, 10);
        assert!(tok.vocab_size() > 4); // at least a,b,c,d + some merges
        assert!(tok.vocab_size() <= 10);
    }

    // BPE 编码和解码必须保持文本语义，否则上层训练样本会被静默污染。
    #[test]
    fn test_bpe_roundtrip() {
        let text = "the quick brown fox jumps over the lazy dog";
        let tok = BpeTokenizer::train(text, 50);
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    // 重复片段应该被合并压缩，这是 BPE 对小语料最直接的收益。
    #[test]
    fn test_bpe_compression() {
        let text = "hello hello hello hello hello world world world";
        let tok = BpeTokenizer::train(text, 30);
        let ids = tok.encode(text);
        assert!(ids.len() < text.len());
    }

    // 明确检查常见 pair merge，防止训练循环只扩词表却没有真正应用合并规则。
    #[test]
    fn test_bpe_vocab_contains_merges() {
        let text = "abababab";
        let tok = BpeTokenizer::train(text, 5);
        assert!(tok.vocab.iter().any(|s| s == "ab"));
    }
}
