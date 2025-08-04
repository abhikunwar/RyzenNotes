import os
from langchain_core.embeddings import Embeddings 
from transformers import AutoTokenizer,AutoModel
import torch

class custom_embedding(Embeddings):
    def __init__(self):
        self.model_name = "BAAI/bge-large-en-v1.5"
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def convert_onnx(self):
        dummy_text = "exporting to onnx" * 64
        dummy_inputs = self.tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length = 512,
            padding="max_length",
            truncation = True)
        
        torch.onnx.convert(
            self.model,
            (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
            "bge-large-en-v1.5-embedding.onnx",
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state","pooler_output"],
            dynamic_axes = None,
            opset_version = 17
        )

            
