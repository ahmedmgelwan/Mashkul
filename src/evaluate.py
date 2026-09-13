import torch
from torch.utils.data import DataLoader

from src.tokenizer import ArabTokenizer


def evaluate(model: torch.nn.Module, loader: DataLoader, metric, device: str) -> float:
    model.eval()
    metric.reset()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            metric.update(y_pred, y_batch)
    return metric.compute().item()


def evaluate_arabic_diacritization(
    model: torch.nn.Module,
    test_loader: DataLoader,
    tokenizer: ArabTokenizer,
    device: str,
    ):
    """
    تقييم شامل بمقاييس متخصصة في التشكيل:
    - WER: نسبة الكلمات الغلط
    - DER: نسبة الحركات الغلط (شامل/بدون آخر حرف في كل كلمة)
    - Exact Match: نسبة الجمل اللي طلعت مطابقة تمامًا
    """
    model.eval()

    total_words = wrong_words = 0
    total_diacritics = wrong_diacritics = 0
    total_diacritics_no_last = wrong_diacritics_no_last = 0
    exact_match_sentences = total_sentences = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch).argmax(dim=1)

            for i in range(x_batch.size(0)):
                mask = x_batch[i] != 0
                x_seq = x_batch[i][mask]
                y_true_seq = y_batch[i][mask]
                y_pred_seq = preds[i][mask]

                true_text = tokenizer.detokenize(x_seq, y_true_seq)
                pred_text = tokenizer.detokenize(x_seq, y_pred_seq)

                total_sentences += 1
                if true_text == pred_text:
                    exact_match_sentences += 1

                for tw, pw in zip(true_text.split(), pred_text.split()):
                    total_words += 1
                    if tw != pw:
                        wrong_words += 1

                for t_id, p_id in zip(y_true_seq, y_pred_seq):
                    total_diacritics += 1
                    if t_id != p_id:
                        wrong_diacritics += 1

                if len(y_true_seq) > 1:
                    for t_id, p_id in zip(y_true_seq[:-1], y_pred_seq[:-1]):
                        total_diacritics_no_last += 1
                        if t_id != p_id:
                            wrong_diacritics_no_last += 1

    results = {
        "wer": (wrong_words / total_words) * 100 if total_words else 0.0,
        "der": (wrong_diacritics / total_diacritics) * 100 if total_diacritics else 0.0,
        "der_no_last": (
            (wrong_diacritics_no_last / total_diacritics_no_last) * 100
            if total_diacritics_no_last
            else 0.0
        ),
        "sentence_accuracy": (
            (exact_match_sentences / total_sentences) * 100 if total_sentences else 0.0
        ),
    }

    print("---  (Evaluation Results) ---")
    print(f"Word Error Rate (WER): {results['wer']:.2f}%")
    print(f"Diacritic Error Rate (DER - شامل الأخير): {results['der']:.2f}%")
    print(f"Diacritic Error Rate (DER - بدون الحرف الأخير): {results['der_no_last']:.2f}%")
    print(f"Exact Match (Sentence Accuracy): {results['sentence_accuracy']:.2f}%")

    return results