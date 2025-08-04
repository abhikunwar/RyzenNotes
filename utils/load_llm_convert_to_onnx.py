from langchain_core.language_models.llms import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Optional, List
from pydantic import PrivateAttr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from langchain_core.language_models.llms import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Optional, List
from pydantic import PrivateAttr


class Custom_LLM(LLM):
    # Class-level storage for shared resources
    _model_instance = None
    _tokenizer_instance = None
    
    def __init__(self, model_dir: str = "../models/phi2", **kwargs):
        super().__init__(**kwargs)
        self.model_dir = model_dir
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model and tokenizer exactly once"""
        if Custom_LLM._model_instance is None:
            try:
                Custom_LLM._tokenizer_instance = AutoTokenizer.from_pretrained(
                    self.model_dir,
                    local_files_only=True
                )
                Custom_LLM._model_instance = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                    
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "phi2-local"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        # Verify initialization
        if Custom_LLM._tokenizer_instance is None or Custom_LLM._model_instance is None:
            self._initialize_model()
        
        # Set generation parameters
        generation_config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 150),
            "temperature": kwargs.get("temperature", 0.5),
            "top_p": kwargs.get("top_p", 0.95),
            "do_sample": kwargs.get("do_sample", True),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        }

        # # Handle stop tokens
        # if stop is not None:
        #     stop_token_ids = []
        #     for stop_word in stop:
        #         tokens = Custom_LLM._tokenizer_instance.encode(stop_word, add_special_tokens=False)
        #         if tokens:
        #             stop_token_ids.append(tokens[0])
        #     if stop_token_ids:
        #         generation_config["eos_token_id"] = stop_token_ids

        # Tokenize input
        inputs = Custom_LLM._tokenizer_instance(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Generate output
        with torch.no_grad():
            outputs = Custom_LLM._model_instance.generate(
                input_ids=input_ids,
                **generation_config
            )

        # Decode and clean output
        output_text = Custom_LLM._tokenizer_instance.decode(outputs[0], skip_special_tokens=True)
        return output_text[len(prompt):] if output_text.startswith(prompt) else output_text.strip()

    # @classmethod
    # def clear_resources(cls):
    #     """Clean up resources"""
    #     cls._model_instance = None
    #     cls._tokenizer_instance = None
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    

if __name__=="__main__":
    llm = Custom_LLM(model_dir="../models/phi2")  # You can omit path if using default

    # Generate text
    response = llm.invoke("Explain quantum computing in simple terms", max_new_tokens=200)
    print(response)
    