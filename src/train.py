import os

import torch
import torch.nn as nn
import torchmetrics
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.config import DEVICE, PathConfig, data_config, model_config, train_config
from src.dataset import TashkeelDataset, collate_fn
from src.evaluate import evaluate, evaluate_arabic_diacritization
from src.model import DiacritizationModelBiGRU
from src.tokenizer import ArabTokenizer


def train(model, train_loader, val_loader, criterion, optimizer, n_epochs, metric, device):
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=train_config.lr_patience, factor=train_config.lr_factor
    )
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        metric.reset()

        for index, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            metric.update(y_pred, y_batch)
            print(f"\rBatch {index + 1}/{len(train_loader)}, loss={total_loss / (index + 1):.4f}", end="")

        train_metric = metric.compute().item()
        val_metric = evaluate(model, val_loader, metric, device)
        scheduler.step(val_metric)

        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(train_metric)
        history["valid_metrics"].append(val_metric)

        print(
            f"\rEpoch {epoch + 1}/{n_epochs}  "
            f"train loss: {history['train_losses'][-1]:.4f}  "
            f"train acc: {train_metric:.2%}  "
            f"val acc: {val_metric:.2%}"
        )

    return history


def main():
    torch.manual_seed(train_config.seed)
    os.makedirs(PathConfig.assets_dir, exist_ok=True)

    print(f"Device: {DEVICE}")

    # 1. تحميل البيانات (streaming عشان الداتاسيت كبير)
    dataset = load_dataset(data_config.dataset_name, streaming=True)
    train_data = dataset["train"].take(data_config.train_size)
    val_data = dataset["dev"].take(data_config.val_size)
    test_data = dataset["test"].take(data_config.test_size)

    # 2. بناء التوكنايزر من بيانات التدريب
    print("Building tokenizer...")
    tokenizer = ArabTokenizer(train_data)
    print(f"Vocab size: {tokenizer.vocab_size} | Labels size: {tokenizer.num_labels}")

    # ملحوظة: لازم نعيد أخذ التريننج داتا تاني لأن الـ streaming iterator اتستهلك فوق
    train_data = dataset["train"].take(data_config.train_size)

    # 3. بناء الداتاسيتس والداتالودرز
    print("Building datasets...")
    train_dataset = TashkeelDataset(train_data, tokenizer, data_config.window_size)
    val_dataset = TashkeelDataset(val_data, tokenizer, data_config.window_size)
    test_dataset = TashkeelDataset(test_data, tokenizer, data_config.window_size)

    train_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_config.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_config.batch_size, shuffle=False, collate_fn=collate_fn
    )
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)} | Test size: {len(test_dataset)}")
    print('Moel and training configuration:')
    # 4. الموديل + التدريب
    model = DiacritizationModelBiGRU.from_config(
        model_config, vocab_size=tokenizer.vocab_size, labels_size=tokenizer.num_labels
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(model.parameters())
    accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=tokenizer.num_labels, ignore_index=0
    ).to(DEVICE)

    print("Starting training...")
    train(model, train_loader, val_loader, criterion, optimizer, train_config.n_epochs, accuracy, DEVICE)

    # 5. تقييم نهائي
    test_acc = evaluate(model, test_loader, accuracy, DEVICE)
    print(f"Test accuracy: {test_acc:.2%}")
    evaluate_arabic_diacritization(model, test_loader, tokenizer, DEVICE)

    # 6. حفظ الموديل والتوكنايزر
    torch.save(model.state_dict(), Paths.MODEL_PATH)
    tokenizer.save(Paths.tokenizer_path)
    print(f"Saved model to {Paths.model_path}")
    print(f"Saved tokenizer to {Paths.tokenizer_path}")


if __name__ == "__main__":
    main()