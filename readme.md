# 情感分类模型训练与推理

## 文件说明
- `new_train.py` - 模型训练脚本
- `inference.py` - 模型推理脚本

## 模型概述
基于多模态大模型先验知识的BERT中文情感分类模型，识别7种情感：
- 正面：like, happiness, surprise
- 负面：sadness, anger, disgust, fear

## 训练流程

### 数据准备
**数据源：**
- 旧数据：[下载链接](link)，下载后，将指定存放路径替换`new_train.py`中的`old_data_dir`变量
- 新数据：[下载链接](link)，下载后，将指定存放路径替换`new_train.py`中的`new_data_root`变量

**数据列格式：**
- `processed_text` - 处理后的文本
- `general_knowledge` - 通用知识
- `domain_knowledge` - 领域知识
- `emotion` - 情感标签（需为上述7种之一）

> 下载后的数据为符合上述格式的数据

### 训练

#### 训练指令
```bash
python new_train.py
```

#### 输出
- 模型保存至 `./emotion_model`目录

- 训练过程显示各epoch的loss和验证集性能


### 推理

#### 前提条件

- 模型路径: 确保 ./emotion_model 目录存在并包含训练好的模型文件

- 多模态大模型服务：需要运行一个兼容 OpenAI 接口的多模态大模型服务（例如 Qwen2.5-VL-7B-Instruct），并且配置正确的 base_url 和 port（默认 http://127.0.0.1:30000/v1）

- 将待推理的数据文件路径替换`inference.py`中的`csv_file_path`和`image_dir_path`变量

#### 推理指令
```bash
python inference.py
```

#### 输出
- 帖子级别的详细预测结果：post_results.csv
- 事件级别的汇总统计：event_summary.json

