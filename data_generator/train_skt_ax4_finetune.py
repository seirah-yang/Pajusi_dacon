# train_skt_ax4_finetune.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

model_name = "skt/ko-gpt-trinity-1.2B-v0.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

data_files = "/home/alpaco/homework/paju-dolbomon/paju_cleaned_sqdm.jsonl"
dataset = load_dataset("json", data_files=data_files)

def tokenize(batch):
    tokenized = tokenizer(
        [f"질문: {i}\n답변: {o}" for i, o in zip(batch["instruction"], batch["output"])],
        truncation=True, padding="max_length", max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

args = TrainingArguments(
    output_dir="/home/alpaco/homework/paju-dolbomon/paju_ax4_finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=3e-5,
    logging_steps=50,
    save_strategy="epoch",
)
## epochs = 3인 이유 -> 필요 시 EarlyStoppingCallback으로 조기 종료하는 것이 과학적·실무적으로 이상적
"""
"We recommend starting with 2–3 epochs for fine-tuning large language models.
Longer training runs often lead to overfitting,
as pre-trained models already have strong language priors."
— Hugging Face Docs: https://huggingface.co/docs/transformers/main_classes/trainer # TrainingArguments"""

trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"])
trainer.train()

model.save_pretrained("/home/alpaco/homework/paju-dolbomon/paju_ax4_finetuned")
tokenizer.save_pretrained("/home/alpaco/homework/paju-dolbomon/paju_ax4_finetuned")

print("Fine-tuning complete. Model saved to /home/alpaco/homework/paju-dolbomon/paju_ax4_finetuned")