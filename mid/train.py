import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
import json

LABEL_MAP = {
    "like": 0, "happiness": 1, "sadness": 2,
    "anger": 3, "disgust": 4, "fear": 5, "surprise": 6
}

def set_seed(seed=2023):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EmotionDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_len=256):
        self.data_path = os.path.join(data_dir, f"{split}_augmented.txt")
        self.data = self._load_data()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        content = sample['content']
        label = sample['label']
        general_knowledge = sample.get('general_knowledge', '')
        domain_knowledge = sample.get('domain_knowledge', '')

        # 拼接提示语言
        enhanced_input = f"{content} [SEP] 常识：{general_knowledge} [SEP] 专业知识：{domain_knowledge}"
        label_id = LABEL_MAP[label]

        encoding = self.tokenizer(
            enhanced_input,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label_id, dtype=torch.long)
        }

def train_model(data_dir):
    """训练情感分类模型"""
    # ========== 模型配置 ==========
    # 基础配置
    model_name = 'bert-base-chinese'
    max_len = 256
    
    # 训练参数（经过验证的最佳参数）
    batch_size = 32
    epochs = 20
    learning_rate = 2e-5
    weight_decay = 0.001
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    
    # 早停参数
    patience = 3
    min_delta = 0.001
    
    # 随机种子
    seed = 77
    
    # ========== 初始化 ==========
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    print(f"随机种子: {seed}")
    
    # ========== 加载模型和tokenizer ==========
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)
    model.to(device)
    
    # ========== 准备数据 ==========
    train_dataset = EmotionDataset(data_dir, "trainval", tokenizer, max_len)
    test_dataset = EmotionDataset(data_dir, "test", tokenizer, max_len)
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # ========== 优化器和调度器 ==========
    # 对bias和LayerNorm参数不应用权重衰减
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # ========== 损失函数 ==========
    # 使用标准交叉熵损失（在这个任务上表现更好）
    loss_fn = nn.CrossEntropyLoss()
    
    # ========== 训练循环 ==========
    best_acc = 0
    best_f1 = 0
    counter = 0
    
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"训练轮次: {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # 训练阶段
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc="训练进度")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 计算损失
            loss = loss_fn(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # 优化步骤
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                '损失': f"{loss.item():.4f}",
                '学习率': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"平均训练损失: {avg_train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="验证进度"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        val_accuracies.append(accuracy)
        val_f1_scores.append(f1)
        
        print(f"\n验证结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 打印分类报告
        print("\n详细分类报告:")
        print(classification_report(
            all_labels, all_preds, 
            target_names=list(LABEL_MAP.keys()), 
            digits=4
        ))
        
        # 检查是否保存最佳模型
        improvement = accuracy - best_acc
        
        if improvement > min_delta:
            best_acc = accuracy
            best_f1 = f1
            counter = 0
            
            # 保存最佳模型
            save_path = os.path.join(data_dir, "bert_emotion_best")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # 保存训练历史
            history = {
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'val_f1_scores': val_f1_scores,
                'best_epoch': epoch + 1,
                'best_accuracy': best_acc,
                'best_f1': best_f1,
                'config': {
                    'model_name': model_name,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'warmup_ratio': warmup_ratio,
                    'max_grad_norm': max_grad_norm,
                    'patience': patience,
                    'min_delta': min_delta,
                    'seed': seed,
                    'max_len': max_len
                }
            }
            
            history_path = os.path.join(data_dir, "training_history.json")
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 保存最佳模型到: {save_path}")
            print(f"✅ 保存训练历史到: {history_path}")
        else:
            counter += 1
            print(f"未改进，早停计数器: {counter}/{patience}")
            
            if counter >= patience:
                print(f"⏹️ 早停触发，停止训练")
                break
    
    # ========== 训练总结 ==========
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    print(f"最佳准确率: {best_acc:.4f}")
    print(f"最佳F1分数: {best_f1:.4f}")
    print(f"总训练轮次: {len(train_losses)}")
    print(f"模型保存位置: {os.path.join(data_dir, 'bert_emotion_best')}")
    
    return best_acc, best_f1


if __name__ == "__main__":
    data_dir = "CLUEdataset/emotion"
    
    # 确保输出目录存在
    os.makedirs(data_dir, exist_ok=True)
      
    # 开始训练
    train_model(data_dir)