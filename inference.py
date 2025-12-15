import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import openai
import base64
import json
from tqdm import tqdm

class EmotionInference:
    def __init__(self, model_path=None, batch_size=32, **kwargs):
        """
        初始化函数，加载模型权重和设置推理参数
        """
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or './emotion_model'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path).to(self.device)
        self.model.eval()
        self.target_emotions = ["like", "happiness", "sadness", "anger", "disgust", "fear", "surprise"]

        port = kwargs.get('port', 30000)
        self.client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

        print("Initialization successful")

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def query_general_knowledge(self, comment=None, image_path=None):
        contents = []
        if comment:
            contents.append({"type": "text", "text": f"Briefly provide concise background knowledge (1-3 sentences only):\n\"{comment}\""})
        if image_path and os.path.exists(image_path):
            base64_image = self.encode_image_to_base64(image_path)
            contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

        response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{"role": "user", "content": contents}],
            temperature=0,
            max_tokens=128
        )
        return response.choices[0].message.content

    def query_domain_knowledge(self, comment=None, image_path=None):
        contents = []
        if comment:
            contents.append({"type": "text", "text": f"Briefly explain (in 1-3 sentences) how people from different cultural backgrounds might interpret or react:\n\"{comment}\""})
        if image_path and os.path.exists(image_path):
            base64_image = self.encode_image_to_base64(image_path)
            contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

        response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{"role": "user", "content": contents}],
            temperature=0,
            max_tokens=128
        )
        return response.choices[0].message.content

    def forward(self, event_name, csv_file_path, image_dir_path):
        df = pd.read_csv(csv_file_path)
        results = []

        positive_emotions = {"like", "happiness", "surprise"}
        negative_emotions = {"sadness", "anger", "disgust", "fear"}

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing posts"):
            text = row.get("全文内容", "") # 全文内容 or 微博正文
            post_id = str(row.get("id", idx))
            image_path = os.path.join(image_dir_path, f"{post_id}.jpg")
            if not os.path.exists(image_path):
                image_path = None

            general_knowledge = self.query_general_knowledge(comment=text, image_path=image_path)
            domain_knowledge = self.query_domain_knowledge(comment=text, image_path=image_path)

            # 拼接输入
            model_input = f"{text} {general_knowledge} {domain_knowledge}"
            encoding = self.tokenizer(model_input, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred_label = torch.argmax(outputs.logits, dim=1).item()
                pred_emotion = self.target_emotions[pred_label]

            results.append({
                "id": post_id,
                "text": text,
                "general_knowledge": general_knowledge,
                "domain_knowledge": domain_knowledge,
                "predicted_emotion": pred_emotion
            })

        # 保存帖子级结果CSV
        post_csv_path = f"{event_name}_post_results.csv"
        pd.DataFrame(results).to_csv(post_csv_path, index=False, encoding='utf-8-sig')

        # 事件级统计
        df_results = pd.DataFrame(results)
        total_posts = len(df_results)
        emotion_counts = df_results["predicted_emotion"].value_counts().reindex(self.target_emotions, fill_value=0)
        emotion_percent = (emotion_counts / total_posts * 100).round(2).to_dict()

        pos_count = df_results["predicted_emotion"].isin(positive_emotions).sum()
        neg_count = df_results["predicted_emotion"].isin(negative_emotions).sum()
        pos_percent = round(pos_count / total_posts * 100, 2)
        neg_percent = round(neg_count / total_posts * 100, 2)

        event_summary = {
            "total_posts": total_posts,
            "emotion_counts": emotion_counts.to_dict(),
            "emotion_percent": emotion_percent,
            "positive": {"count": int(pos_count), "percent": pos_percent},
            "negative": {"count": int(neg_count), "percent": neg_percent}
        }

        event_json_path = f"{event_name}_event_summary.json"
        with open(event_json_path, "w", encoding="utf-8") as f:
            json.dump(event_summary, f, ensure_ascii=False, indent=4)

        return event_json_path, post_csv_path


if __name__ == "__main__":
    inference_model = EmotionInference(model_path="./emotion_model", batch_size=1, port=30000)

    event_name = "女子称在三亚冲浪时遭教练猥亵"
    event_json, post_csv = inference_model.forward(
        event_name=event_name,
        csv_file_path=f"/network_space/server129/lttt/PublicOpinionData/data/结果文件/{event_name}/{event_name}.csv",
        image_dir_path=f"/network_space/server129/lttt/PublicOpinionData/data/结果文件/{event_name}/images"
    )

    print("事件级结果:", event_json)
    print("帖子级结果:", post_csv)
