from typing import List, Tuple

import torch

from src.config import data_config
from src.model import DiacritizationModelBiGRU
from src.tokenizer import ArabTokenizer


class Diacritizer:
    def __init__(self, model: DiacritizationModelBiGRU, tokenizer: ArabTokenizer, device: str):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(cls, model_path: str, tokenizer_path: str, device: str) -> "Diacritizer":
        tokenizer = ArabTokenizer.load(tokenizer_path)
        model = DiacritizationModelBiGRU(
            vocab_size=tokenizer.vocab_size,
            labels_size=tokenizer.num_labels,
        )
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        return cls(model, tokenizer, device)

    def diacritize(self, text: str, window_size: int = data_config.window_size) -> str:
        chars, _ = self.tokenizer._extract_chars_and_labels(text)
        clean_text = "".join(chars)
        diacritizable_positions, char_ids = self._split_known_chars(clean_text)

        full_pred_labels: List[int] = []
        with torch.no_grad():
            for i in range(0, len(char_ids), window_size):
                chunk = char_ids[i : i + window_size]
                x = torch.tensor([chunk], dtype=torch.long, device=self.device)
                logits = self.model(x)
                preds = logits.argmax(dim=1)
                full_pred_labels.extend(preds[0].cpu().tolist())

        result = list(clean_text)
        for pos, label_id in zip(diacritizable_positions, full_pred_labels):
            label = self.tokenizer.index_to_label.get(label_id, "")
            if label == "<PAD>":  # نظريًا نادر جدًا مع موديل مدرب، لكن لازم نحميه
                label = ""
            result[pos] = result[pos] + label
        print(len("".join(result)))
        return "".join(result)

    def _split_known_chars(self, text: str) -> Tuple[List[int], List[int]]:
        """بيفصل مواقع الحروف اللي الموديل عارفها عن باقي النص (مسافات/ترقيم/غيره)."""
        positions: List[int] = []
        char_ids: List[int] = []
        for i, ch in enumerate(text):
            if ch in self.tokenizer.char_to_index:
                positions.append(i)
                char_ids.append(self.tokenizer.char_to_index[ch])
        return positions, char_ids