import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.tokenizer import ArabTokenizer


class TashkeelDataset(Dataset):
    def __init__(self, data, tokenizer: ArabTokenizer, window_size: int):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.inputs: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

        for row_no, row in enumerate(data):
            char_ids, label_ids = self.tokenizer.tokenize(row["diacratized"])

            for i in range(0, len(char_ids), self.window_size):
                self.inputs.append(char_ids[i : i + self.window_size])
                self.labels.append(label_ids[i : i + self.window_size])

            if row_no % 1000 == 0:
                print(f"\rtokenizing row {row_no}", end="")
        print()

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.labels[index]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return padded_inputs, padded_labels