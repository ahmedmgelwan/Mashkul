import torch
import json

class ArabTokenizer:
    def __init__(self, data_iterator=None):
        self._diacritics = ['َ', 'ً', 'ُ', 'ٌ', 'ِ', 'ٍ', 'ْ', 'ّ']

        self.char_to_index = {}
        self.index_to_char = {}
        self.label_to_index = {}
        self.index_to_label = {}

        if data_iterator is not None:
            self._build_vocab(data_iterator)

    def _build_vocab(self, data_iterator):
        unique_chars = set()
        unique_labels = set()

        for row in data_iterator:
            chars, labels = self._extract_chars_and_labels(row['diacratized'])
            unique_chars.update(chars)
            unique_labels.update(labels)

        if "" in unique_labels:
            unique_labels.remove("")

        self.chars = ['<PAD>', '<UNK>'] + sorted(list(unique_chars))
        self.char_to_index = {char: idx for idx, char in enumerate(self.chars)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.chars)}

        self.labels = ['<PAD>', ''] + sorted(list(unique_labels))
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(self.labels)}

    def _extract_chars_and_labels(self, text):
        chars = []
        labels = []
        for char in text:
            if char not in self._diacritics:
                chars.append(char)
                labels.append('')
            else:
                if len(labels) > 0:
                    labels[-1] += char
                    # Normalize diacritic combinations (e.g., ensure Shadda comes first if present)
                    if len(labels[-1]) == 2 and labels[-1][1] == 'ّ':
                        labels[-1] = 'ّ' + labels[-1][0]
        return chars, labels

    def tokenize(self, text):
        chars, labels = self._extract_chars_and_labels(text)

        char_ids = [self.char_to_index.get(char, 1) for char in chars]
        label_ids = [self.label_to_index.get(label, 1) for label in labels]

        return torch.tensor(char_ids), torch.tensor(label_ids)

    def detokenize(self, char_indexes, label_indexes):
        result = []
        for c_idx, l_idx in zip(char_indexes, label_indexes):
            char = self.index_to_char.get(c_idx.item(), '<UNK>')
            label = self.index_to_label.get(l_idx.item(), '')
            if char not in ['<PAD>', '<UNK>']:
                result.append(char + label)
        return "".join(result)

    def save(self, path):
        data = {
            'chars': self.chars,
            'labels': self.labels
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.chars = data['chars']
        tokenizer.labels = data['labels']

        tokenizer.char_to_index = {char: idx for idx, char in enumerate(tokenizer.chars)}
        tokenizer.index_to_char = {idx: char for idx, char in enumerate(tokenizer.chars)}

        tokenizer.label_to_index = {label: idx for idx, label in enumerate(tokenizer.labels)}
        tokenizer.index_to_label = {idx: label for idx, label in enumerate(tokenizer.labels)}
        return tokenizer

    @property
    def vocab_size(self):
        return len(self.chars)

    @property
    def num_labels(self):
        return len(self.labels)
    