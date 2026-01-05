# 情感分析模型训练与推理


## 模型概述

本模型是一个基于多模态大语言模型（MLLM）增强知识的中文情感分类模型。模型通过融合大模型提供的通用知识与领域知识，实现对社交媒体帖子（文本或文本+图像）的细粒度情感分析

### 核心功能

- 算法输入：社交媒体帖子（纯文本或文本+图像）
- 算法输出：7种情感类别分类结果

- 分类体系：
    - 正面情感：喜欢（like）、快乐（happiness）、惊讶（surprise）
    - 负面情感：悲伤（sadness）、愤怒（anger）、厌恶（disgust）、恐惧（fear）

### 技术亮点

通过多模态大模型获取通用知识（[代码链接](https://github.com/ltttpku/Post_Emotion_Analysis/blob/main/inference.py#L32)）与领域知识（[代码链接](https://github.com/ltttpku/Post_Emotion_Analysis/blob/main/inference.py#L48)），提升情感分析准确率

## 训练流程

### 数据准备
**数据源：**
- 旧数据：[下载链接](https://disk.pku.edu.cn/link/AA949EA10AD1FE438781190519A43F0129)
    - 下载后，将指定存放路径替换`new_train.py`中的`old_data_dir`变量
- 新数据：[下载链接](https://disk.pku.edu.cn/link/AA949EA10AD1FE438781190519A43F0129)
    - 下载后，将指定存放路径替换`new_train.py`中的`new_data_root`变量

**数据列格式：**
- `processed_text` - 预处理后的文本内容
- `general_knowledge` - 多模态大模型提取的通用知识
- `domain_knowledge` - 多模态大模型提取的领域知识
- `emotion` - 情感标签（必须为上述7种情感之一）

> 注：下载的数据应为符合上述格式的数据

### 训练

#### 训练指令
```bash
python new_train.py
```

#### 输出
- 模型保存路径： `./emotion_model`目录

- 训练监控：控制台显示各epoch的loss和验证集性能


## 推理流程

### 环境准备

- 模型准备: 
    - 确保 `./emotion_model` 目录存在并包含训练好的模型文件
    - 我们提供训练好的情感分析模型：[下载链接](https://disk.pku.edu.cn/link/AA1BCDDD0F4AEE4CB3A8A11C8515615D9B)

- 多模态大模型服务准备：
    - 需要运行一个兼容 OpenAI 接口的多模态大模型服务（例如 `Qwen2.5-VL-7B-Instruct`），并且配置正确的 `base_url` 和 `port`（默认 http://127.0.0.1:30000/v1 ）

- 推理数据准备：
    - 将待推理的数据文件路径替换`inference.py`中的`csv_file_path`和`image_dir_path`变量

### 推理指令
```bash
python inference.py
```

#### 输出
- 帖子级别的详细预测结果：`post_results.csv`
- 事件级别的汇总统计：`event_summary.json`

