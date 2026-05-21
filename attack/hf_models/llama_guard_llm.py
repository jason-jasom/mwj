import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlamaGuardLLM:
    """Llama Guard 3 8B wrapper that uses apply_chat_template() as the model expects."""

    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        print(f"[LlamaGuardLLM] Loading {model_name} with apply_chat_template support")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def generate(self, full_prompt, max_new_tokens, temperature, top_p):
        # full_prompt is a JSON-serialized list of {"role": ..., "content": ...} dicts
        try:
            messages = json.loads(full_prompt)
        except (json.JSONDecodeError, TypeError):
            messages = [{"role": "user", "content": full_prompt}]

        print(f"[LlamaGuardLLM] Generating with messages: {messages} and max_new_tokens={max_new_tokens}")

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.llm.device)
        inputs_len = inputs["input_ids"].shape[1]

        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1,
            top_p=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(outputs[0][inputs_len:], skip_special_tokens=True)

        for key in inputs:
            inputs[key].to("cpu")
        outputs.to("cpu")
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

        return response
