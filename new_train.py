import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import random

# -------------------- 配置 --------------------
pretrained_model = "bert-base-chinese"  # 若是英文数据可改为 bert-base-uncased
output_dir = "./emotion_model"
seed = 42

target_emotions = [
    "like", "happiness", "sadness", "anger",
    "disgust", "fear", "surprise"
]

# 固定随机种子
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# -------------------- 数据集类 --------------------
class EmotionDataset(Dataset):
    def __init__(self, csv_files, tokenizer, max_len=256):
        self.samples = []
        self.all_labels = []
        for file in csv_files:
            df = pd.read_csv(file)
            df = df[df["emotion"].isin(target_emotions)]
            for col in ["processed_text", "general_knowledge", "domain_knowledge"]:
                if col in df.columns:
                    df[col] = df[col].fillna("").astype(str)
                else:
                    df[col] = ""
            self.all_labels.extend(df["emotion"].tolist())
            for _, row in df.iterrows():
                text = f"{row['processed_text']} {row['general_knowledge']} {row['domain_knowledge']}".strip()
                label = target_emotions.index(row["emotion"])
                self.samples.append((text, label))

        self._balance_labels(max_per_class=40000)

        distribution = pd.Series(self.all_labels).value_counts()
        print("Emotion distribution:")
        print(distribution)

        self.tokenizer = tokenizer
        self.max_len = max_len
    
    
    def _balance_labels(self, max_per_class=30000):
        """对每个 label 执行下采样，使其数量不超过 max_per_class"""

        from collections import defaultdict
        import random

        label_buckets = defaultdict(list)

        # 将样本按 label 分桶
        for idx, (text, label) in enumerate(self.samples):
            label_buckets[label].append((text, label))

        new_samples = []

        for label, items in label_buckets.items():
            if len(items) > max_per_class:
                # 下采样大类
                new_items = random.sample(items, max_per_class)
            else:
                # 小类不处理
                new_items = items
            new_samples.extend(new_items)

        random.shuffle(new_samples)
        self.samples = new_samples
        self.all_labels = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# -------------------- 数据准备 --------------------

# （1）老数据路径
old_data_dir = "/hd1/shared/leiting/yuqing/annotation/cleaned_data_emotion"
old_csv_files = [
    os.path.join(old_data_dir, f) 
    for f in os.listdir(old_data_dir) 
    if f.endswith(".csv")
]

# （2）新数据路径（你生成的 emotion.csv）
new_data_root = "/network_space/server126/shared/leiting126/midudata/events"
new_csv_files = []
for event_folder in os.listdir(new_data_root):
    event_path = os.path.join(new_data_root, event_folder)
    if not os.path.isdir(event_path):
        continue

    emotion_csv = os.path.join(event_path, "emotion.csv")
    if os.path.exists(emotion_csv):
        new_csv_files.append(emotion_csv)

print(f"发现旧数据 {len(old_csv_files)} 个文件")
print(f"发现新数据 {len(new_csv_files)} 个文件")

csv_files = old_csv_files + new_csv_files
assert len(csv_files) > 0, "未找到任何 CSV 文件，请检查路径"


# -------- 初始化 tokenizer + Dataset ----------
tokenizer = BertTokenizer.from_pretrained(pretrained_model)
all_dataset = EmotionDataset(csv_files, tokenizer)

# -------- 划分训练集 / 验证集 ----------
split = int(0.8 * len(all_dataset))
train_dataset, eval_dataset = random_split(
    all_dataset, [split, len(all_dataset) - split], generator=torch.Generator().manual_seed(seed)
)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16)

num_labels = len(target_emotions)

# -------------------- 模型 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels).to(device)

# -------------------- 损失函数 --------------------
loss_fn = torch.nn.CrossEntropyLoss()

# -------------------- 优化器 --------------------
optimizer = AdamW(model.parameters(), lr=2e-5)

# -------------------- 训练 --------------------
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

    # 验证
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="weighted")
    print(f"Validation Accuracy: {acc:.4f}, Weighted F1: {f1:.4f}")
    print(classification_report(trues, preds, target_names=target_emotions))

# -------------------- 保存 --------------------
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
