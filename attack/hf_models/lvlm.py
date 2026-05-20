import torch
import gc
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    AutoTokenizer,
    AutoProcessor,
    MllamaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    MllamaForConditionalGeneration,
)


class HuggingFaceLVLM:
    def __init__(self, model_name: str, model_path: str):
        if "llava" in model_name:
            self.lvlm = LlavaNextForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.processor = LlavaNextProcessor.from_pretrained(model_path)
        elif model_name == "qwen2":
            self.lvlm = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=640 * 28 * 28)
        elif model_name == "qwen2.5":
            self.lvlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=640 * 28 * 28)
        elif model_name == "qwen3":
            self.lvlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=640 * 28 * 28)
        elif "llama_guard" in model_name:
            self.lvlm = MllamaForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
        elif "llama" in model_name:
            self.lvlm = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            raise NotImplementedError

    def generate(self, conv, images, max_new_tokens, temperature, top_p):
        prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)

        if not images:
            images = None
        print("Images:", len(images) if images else "No images provided")
        print("Text:", prompt)

        # It is safer to move the tensors to model's device key-by-key instead
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.lvlm.device)
        text_prompt_len = inputs.input_ids.shape[-1]

        if temperature > 0:
            outputs = self.lvlm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        else:
            outputs = self.lvlm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1,
                top_p=1,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        response = self.processor.decode(outputs[0][text_prompt_len:], skip_special_tokens=True)

        for key in inputs:
            inputs[key].to("cpu")
        outputs.to("cpu")
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

        return response
