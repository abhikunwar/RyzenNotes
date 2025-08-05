from langchain_core.language_models.llms import LLM
from typing import Any, Optional, List
import onnxruntime_genai as og
from pydantic import PrivateAttr

class Phi3ONNX_LLM(LLM):
    model_path: str = "../cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    max_length: int = 1000
    temperature: float = 0.7

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _tokenizer_stream: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_model()

    def _initialize_model(self):
        self._model = og.Model(self.model_path)
        self._tokenizer = og.Tokenizer(self._model)
        self._tokenizer_stream = self._tokenizer.create_stream()

    @property
    def _llm_type(self) -> str:
        return "phi3-onnx-cpu"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        # Format chat prompt
        chat_prompt = f"<user>\n{prompt}<end>\n<assistant>"
        input_tokens = self._tokenizer.encode(chat_prompt)

        total_max_length = kwargs.get("max_length", self.max_length)
        temperature = kwargs.get("temperature", self.temperature)

        # Reserve space for output tokens
        output_token_budget = 256  # You can tune this
        input_token_limit = total_max_length - output_token_budget

        if len(input_tokens) > input_token_limit:
            print(f"⚠️ Prompt too long ({len(input_tokens)} tokens), truncating to {input_token_limit}")
            input_tokens = input_tokens[:input_token_limit]



        # Set up generation params
        params = og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=total_max_length,
            temperature=temperature
        )

        # Create generator
        generator = og.Generator(self._model, params)
        generator.append_tokens(input_tokens)

        # Generate tokens
        for _ in range(output_token_budget):
            if generator.is_done():
                break
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            print(self._tokenizer_stream.decode(new_token), end='', flush=True)

        print("\n" + "-" * 50)

        # Decode full sequence
        output_text = self._tokenizer.decode(generator.get_sequence(0)[0])

        # Remove prompt if included
        if output_text.startswith(chat_prompt):
            output_text = output_text[len(chat_prompt):]

        # Handle stop words
        # if stop:
        #     for word in stop:
        #         if word in output_text:
        #             output_text = output_text.split(word)[0]

        return output_text.strip()

# Example usage
if __name__ == "__main__":
    llm = Phi3ONNX_LLM()

    print("\n\nAnswer 1:")
    print(llm.invoke("Explain quantum computing in simple terms"))

    print("\n\nAnswer 2:")
    response = llm.invoke(
        "Write a poem about artificial intelligence.",
        temperature=0.9,
        max_length=1000
    )
    print(response)
