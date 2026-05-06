use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;

// ============ Tokenizer Trait ============

pub trait Tokenizer {
    // 中文注释：关键逻辑说明。
    fn encode(&self, text: &str) -> Vec<usize>;
    // 中文注释：关键逻辑说明。
    fn decode(&self, ids: &[usize]) -> String;
    // 中文注释：关键逻辑说明。
    fn vocab_size(&self) -> usize;
}

// ============ CharTokenizer ============

pub struct CharTokenizer {
    pub vocab: Vec<char>,
    char_to_id: HashMap<char, usize>,
}

impl CharTokenizer {
    /// `from_text`：中文注释，说明函数用途、输入约束与输出语义。
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
    // 中文注释：关键逻辑说明。
    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().filter_map(|c| self.char_to_id.get(&c).copied()).collect()
    }

    // 中文注释：关键逻辑说明。
    fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| self.vocab[id]).collect()
    }

    // 中文注释：关键逻辑说明。
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ============ BPE Tokenizer ============

pub struct BpeTokenizer {
    vocab: Vec<String>, // id -> token string
    token_to_id: HashMap<String, usize>,
    merges: Vec<(usize, usize)>, // ordered merge rules: (id_a, id_b) -> new_id
}

impl BpeTokenizer {
    /// `train`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn train(text: &str, target_vocab_size: usize) -> Self {
        // Initialize with single characters found in text
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut vocab: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        let mut token_to_id: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

        // Encode text as initial token ids
        let mut token_ids: Vec<usize> = text
            .chars()
            .map(|c| *token_to_id.get(&c.to_string()).unwrap())
            .collect();

        let mut merges = Vec::new();

        while vocab.len() < target_vocab_size {
            // Count adjacent pairs
            let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
            for w in token_ids.windows(2) {
                *pair_counts.entry((w[0], w[1])).or_insert(0) += 1;
            }

            if pair_counts.is_empty() {
                break;
            }

            // Find most frequent pair
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

            // Apply merge to token_ids
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
    // 中文注释：关键逻辑说明。
    fn encode(&self, text: &str) -> Vec<usize> {
        // Start with character-level tokens
        let mut ids: Vec<usize> = text
            .chars()
            .filter_map(|c| self.token_to_id.get(&c.to_string()).copied())
            .collect();

        // Apply merges in order
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

    // 中文注释：关键逻辑说明。
    fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| self.vocab[id].as_str()).collect()
    }

    // 中文注释：关键逻辑说明。
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ============ Dataset Trait ============

pub trait Dataset {
    // 中文注释：关键逻辑说明。
    fn len(&self) -> usize;
    // 中文注释：关键逻辑说明。
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    // 中文注释：关键逻辑说明。
    fn get(&self, index: usize) -> (Vec<usize>, Vec<usize>);
}

// ============ TextDataset ============

pub struct TextDataset {
    tokens: Vec<usize>,
    seq_len: usize,
}

impl TextDataset {
    /// `new`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn new(tokens: Vec<usize>, seq_len: usize) -> Self {
        TextDataset { tokens, seq_len }
    }
}

impl Dataset for TextDataset {
    // 中文注释：关键逻辑说明。
    fn len(&self) -> usize {
        if self.tokens.len() <= self.seq_len {
            0
        } else {
            self.tokens.len() - self.seq_len
        }
    }

    // 中文注释：关键逻辑说明。
    fn get(&self, index: usize) -> (Vec<usize>, Vec<usize>) {
        let input = self.tokens[index..index + self.seq_len].to_vec();
        let target = self.tokens[index + 1..index + self.seq_len + 1].to_vec();
        (input, target)
    }
}

// ============ DataLoader ============

pub struct DataLoader<'a, D: Dataset> {
    dataset: &'a D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    pos: usize,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    /// `new`：中文注释，说明函数用途、输入约束与输出语义。
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
    /// `reset`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn reset(&mut self) {
        self.pos = 0;
        if self.shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
    }

    /// Returns (inputs, targets) where each is a Vec of sequences.
    #[allow(clippy::type_complexity)]
    /// `next_batch`：中文注释，说明函数用途、输入约束与输出语义。
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
    /// `num_batches`：中文注释，说明函数用途、输入约束与输出语义。
    pub fn num_batches(&self) -> usize {
        self.indices.len().div_ceil(self.batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_char_tokenizer() {
        let tok = CharTokenizer::from_text("hello");
        assert_eq!(tok.vocab_size(), 4); // e, h, l, o
        let ids = tok.encode("hello");
        assert_eq!(ids.len(), 5);
        assert_eq!(tok.decode(&ids), "hello");
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_char_tokenizer_roundtrip() {
        let text = "the quick brown fox";
        let tok = CharTokenizer::from_text(text);
        let ids = tok.encode(text);
        assert_eq!(tok.decode(&ids), text);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_text_dataset() {
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let ds = TextDataset::new(tokens, 4);
        assert_eq!(ds.len(), 4); // 8 - 4 = 4 valid positions
        let (inp, tgt) = ds.get(0);
        assert_eq!(inp, vec![0, 1, 2, 3]);
        assert_eq!(tgt, vec![1, 2, 3, 4]);
    }

    // 中文注释：关键逻辑说明。
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

    // 中文注释：关键逻辑说明。
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

    // 中文注释：关键逻辑说明。
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

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_bpe_roundtrip() {
        let text = "the quick brown fox jumps over the lazy dog";
        let tok = BpeTokenizer::train(text, 50);
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_bpe_compression() {
        let text = "hello hello hello hello hello world world world";
        let tok = BpeTokenizer::train(text, 30);
        let ids = tok.encode(text);
        // BPE should compress repeated patterns, so fewer tokens than chars
        assert!(ids.len() < text.len());
    }

    // 中文注释：关键逻辑说明。
    #[test]
    fn test_bpe_vocab_contains_merges() {
        let text = "abababab";
        let tok = BpeTokenizer::train(text, 5);
        // Should have merged "a"+"b" -> "ab"
        assert!(tok.vocab.iter().any(|s| s == "ab"));
    }
}
