# src/bert_model.py

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss

from config import MODEL_NAME, MAX_LENGTH, BATCH_SIZE, LEARNING_RATE, EPOCHS, RANDOM_SEED


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation_side="left"
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, predictions, average="macro")

    return {
        "macro_f1": macro_f1
    }

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Move weights to same device as logits
        class_weights = self.class_weights.to(logits.device)

        loss_fct = CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def train_bert(train_df, val_df):
    """
    Fine-tune BERT using HuggingFace Trainer API.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Compute class weights
    labels_array = train_df["label"].values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_array),
        y=labels_array,
    )

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Convert pandas → HF Dataset
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]])

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
    )

    training_args = TrainingArguments(
        output_dir="./models",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        seed=RANDOM_SEED,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor.to(model.device),
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model("./models/bert_weighted")

    return trainer


def evaluate_bert(trainer, test_df):
    """
    Evaluate trained BERT model on test set.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    predictions = trainer.predict(test_dataset)

    logits = predictions.predictions
    y_pred = np.argmax(logits, axis=-1)

    return y_pred