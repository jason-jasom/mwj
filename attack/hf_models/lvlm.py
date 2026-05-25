import torch
import gc
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    MllamaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    MllamaForConditionalGeneration,
)


class HuggingFaceLVLM:
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        if "llava" in model_name:
            self.lvlm = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            # patch_size is not in the processor config; pull it from the vision config.
            self.processor.patch_size = self.lvlm.config.vision_config.patch_size
            # LlavaProcessor off-by-one fix:
            #   CLIP outputs (n_patches + 1) hidden states (CLS first, then patches).
            #   The model's "default" strategy does [:, 1:] → keeps all n_patches (576).
            #   But LlavaProcessor.__call__ with "default" does (336//14)**2 - 1 = 575.
            #   Forcing "full" here makes the processor skip the -1 → 576 tokens,
            #   which matches the 576 features the model actually produces.
            self.processor.vision_feature_select_strategy = "full"
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
            self.lvlm = Qwen3VLForConditionalGeneration.from_pretrained(
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

    @staticmethod
    def _flatten_conv(conv) -> list:
        """Convert structured OpenAI-style conv (content as typed-dict list) to the
        flat string format expected by LLaVA: image token embedded inline as '<image>\\n'."""
        flat = []
        for msg in conv:
            text = ""
            has_image = False
            for part in msg["content"]:
                if part["type"] == "text":
                    text = part["text"]
                elif part["type"] == "image":
                    has_image = True
            content = ("<image>\n" if has_image else "") + text
            flat.append({"role": msg["role"], "content": content})
        return flat

    def generate(self, conv, images, max_new_tokens, temperature, top_p):
        if "llava" in self.model_name:
            # Intel/llava-gemma-2b (and similar LLaVA models) require:
            #   1. flat string content with <image>\n inline
            #   2. apply_chat_template on the *tokenizer* (not the processor) with tokenize=False
            flat_conv = self._flatten_conv(conv)
            prompt = self.processor.tokenizer.apply_chat_template(  # type: ignore[union-attr]
                flat_conv, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)  # type: ignore[union-attr]

        if not images:
            images = None
        print("Images:", len(images) if images else "No images provided")
        print("Text:", prompt)

        # LlavaProcessor expects a single PIL Image (not a list) when there is one image.
        # Other processors (Qwen, Llama) handle lists fine, so only unwrap for llava.
        image_input = images
        if "llava" in self.model_name and isinstance(images, list):
            image_input = images[0] if len(images) == 1 else images

        # It is safer to move the tensors to model's device key-by-key instead
        inputs = self.processor(images=image_input, text=prompt, return_tensors="pt").to(self.lvlm.device)
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
        outputs.to("cpu")  # type: ignore[union-attr]
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

        return response
