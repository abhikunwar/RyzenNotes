from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

model_name = "microsoft/phi-2"
save_dir = "../models/phi2_onnx"

# Load original model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Export to ONNX
ort_model = ORTModelForCausalLM.from_pretrained(
    model,
    export=True,
    provider="CPUExecutionProvider",  # Use "CPUExecutionProvider" for CPU
    use_merged=True,
    save_dir=save_dir
)

# Save tokenizer
tokenizer.save_pretrained(save_dir)

print(f"Model successfully converted and saved to {save_dir}")