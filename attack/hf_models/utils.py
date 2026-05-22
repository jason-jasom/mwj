from dotenv import load_dotenv
import os
from enum import Enum
import torch


load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR")

current_dir = os.path.dirname(os.path.abspath(__file__))


def load_model(model_name):
    model_path, model_type = model_name_to_path_type(model_name)
    print(f"Loading model: {model_name}")
    if model_type == ModelType.LLM:
        from .llm import HuggingFaceLLM

        if "strongreject_judge" in model_name:
            from .strongreject import StrongRejectModel

            model = StrongRejectModel(model_name, model_path)
        elif "Llama-Guard" in model_name:
            from .llama_guard_llm import LlamaGuardLLM

            model = LlamaGuardLLM(model_name, model_path)
        else:
            model = HuggingFaceLLM(model_name, model_path)
    elif model_type == ModelType.LVLM:
        from .lvlm import HuggingFaceLVLM

        model = HuggingFaceLVLM(model_name, model_path)
    elif model_type == ModelType.CLIP:
        from .clip import HuggingFaceCLIP

        model = HuggingFaceCLIP(model_path)
    elif model_type == ModelType.SD:
        from .sd import StableDiffusion

        model = StableDiffusion(model_path)
    elif model_type == ModelType.SEM_RELEVANCE:
        from .sem_relvance import SemRelvance

        model = SemRelvance()
    elif model_type == ModelType.TOXIGEN:
        from .toxigen import Toxigen

        model = Toxigen()
    else:
        raise NotImplementedError(f"Unsupported model type: {model_type}")

    print(f"Loaded model: {model_name}")
    return model


class ModelType(Enum):
    LLM = "LLM"
    LVLM = "LVLM"
    CLIP = "CLIP"
    SD = "SD"
    SEM_RELEVANCE = "SEM_RELEVANCE"
    TOXIGEN = "TOXIGEN"


def model_name_to_path_type(name) -> tuple[str, ModelType]:
    look_up = {
        "mistral_7b": {"folder": "mistral_7b_instruct_v0.3", "type": ModelType.LLM},
        "mistral": {"folder": "mistral_small_24b_instruct_2501", "type": ModelType.LLM},
        # "llava": {"folder": "llava_v1.6_mistral_7b_hf", "type": ModelType.LVLM},
        # "qwen": {"folder": "qwen2.5_vl_7b_instruct", "type": ModelType.LVLM},
        # "llama": {"folder": "llama_3.2_11b_vision_instruct", "type": ModelType.LVLM},
        "qwen2.5_instruct": {"folder": "Qwen_Qwen2.5-7B-Instruct", "type": ModelType.LLM},
        "meta-llama_Llama-3.1-8B-Instruct": {"folder": "meta-llama_Llama-3.1-8B-Instruct", "type": ModelType.LLM},
        "meta-llama_Llama-Guard-3-8B": {"folder": "meta-llama_Llama-Guard-3-8B", "type": ModelType.LLM},
        "qwen2": {"folder": "Qwen_Qwen2-VL-2B-Instruct", "type": ModelType.LVLM},
        "qwen2.5": {"folder": "Qwen_Qwen2.5-VL-3B-Instruct", "type": ModelType.LVLM},
        "qwen3": {"folder": "Qwen_Qwen3-VL-4B-Instruct", "type": ModelType.LVLM},
        "llava": {"folder": "Intel_llava-gemma-2b", "type": ModelType.LVLM},
        "clip": {"folder": "clip_vit_base_patch32", "type": ModelType.CLIP},
        "stable_diffusion": {"folder": "stable_diffusion_3.5_medium", "type": ModelType.SD},
        "strongreject_judge": {"folder": "strongreject_15k_v1", "type": ModelType.LLM},
        "harmbench_judge": {"folder": "harmbench_llama_2_13b_cls", "type": ModelType.LLM},
        "cais_HarmBench-Mistral-7b-val-cls": {"folder": "cais_HarmBench-Mistral-7b-val-cls", "type": ModelType.LLM},
        "llama_guard": {"folder": "llama_guard_3_11b_vision", "type": ModelType.LVLM},
        "sem_relevance": {"folder": "", "type": ModelType.SEM_RELEVANCE},
        "toxigen": {"folder": "", "type": ModelType.TOXIGEN},
    }
    path = os.path.join(MODEL_DIR, look_up[name]["folder"])
    type = look_up[name]["type"]

    return path, type
