import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from hf_models.utils import load_model
from typing import List
import json
from datetime import datetime
import logging
import io
import argparse
import asyncio
import gc
import torch
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO)


class LazyModelManager:
    def __init__(self, keep_loaded=False):
        self.current_name = None
        self.current_model = None
        self.keep_loaded = keep_loaded
        self.lock = asyncio.Lock()

    async def run(self, model_name, callback):
        async with self.lock:
            model = self._load_if_needed(model_name)
            try:
                return callback(model)
            finally:
                if not self.keep_loaded:
                    self.unload()

    def unload(self):
        if self.current_model is not None:
            logging.info("Unloading model: %s", self.current_name)
        self.current_name = None
        self.current_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                logging.debug("torch.cuda.ipc_collect() skipped", exc_info=True)

    def _load_if_needed(self, model_name):
        if self.current_name == model_name and self.current_model is not None:
            return self.current_model

        self.unload()
        self.current_name = model_name
        self.current_model = load_model(model_name)
        return self.current_model


def config():
    config = argparse.ArgumentParser()
    config.add_argument("--llm_name", type=str, default="qwen2.5_instruct", choices=["mistral", "qwen2.5_instruct"])
    config.add_argument("--llm_7b_name", type=str, default="mistral_7b")
    config.add_argument("--lvlm_name", type=str, default="llava", choices=["", "llava","qwen2","qwen2.5","qwen3"])
    config.add_argument("--load_clip", action="store_true")
    config.add_argument("--load_sd", action="store_true")
    config.add_argument("--load_sem_relevance", action="store_true")
    config.add_argument("--load_toxigen", action="store_true")
    config.add_argument("--load_llama_guard", action="store_true")
    config.add_argument("--keep_model_loaded", action="store_true")
    config.add_argument("--dataset", type=str, default="")
    config.add_argument("--port", type=int, default=8000)

    return config


def main(args):
    app = FastAPI()
    model_manager = LazyModelManager(keep_loaded=args.keep_model_loaded)
    allowed_llm_models = {
        args.llm_name,
        args.llm_7b_name,
        "meta-llama_Llama-3.1-8B-Instruct",
        "meta-llama_Llama-Guard-3-8B",
        "strongreject_judge",
        "harmbench_judge",
    }

    @app.middleware("http")
    async def log_request_time(request: Request, call_next):
        now = datetime.now().isoformat()
        logging.info(f"{request.method} {request.url} at {now}")
        response = await call_next(request)
        return response

    @app.on_event("shutdown")
    async def unload_model_on_shutdown():
        model_manager.unload()

    async def read_images(files: List[UploadFile] = None):
        images = []
        if files:
            for file in files:
                contents = await file.read()
                try:
                    image = Image.open(BytesIO(contents))
                    images.append(image)
                except Exception as e:
                    return None, {"error": f"Could not process {file.filename}: {e}"}
        return images, None

    @app.get("/llm_gen")
    async def llm_generation(request: Request):
        data = await request.json()
        full_prompt = data.get("full_prompt")
        model_name = data.get("model_name")
        max_new_tokens = data.get("max_new_tokens")
        temperature = data.get("temperature")
        top_p = data.get("top_p")

        if model_name not in allowed_llm_models:
            error_msg = f"Model not found: {model_name}."
            print(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        response = await model_manager.run(
            model_name,
            lambda model: model.generate(full_prompt, max_new_tokens, temperature, top_p),
        )

        return response

    if args.lvlm_name:
        allowed_lvlm_models = {args.lvlm_name}
        if args.load_llama_guard:
            allowed_lvlm_models.add("llama_guard")

        @app.post("/lvlm_gen")
        async def lvlm_generation(files: List[UploadFile] = File(default=None), metadata: str = Form(...)):
            metadata_dict = json.loads(metadata)
            images, error = await read_images(files)
            if error:
                return error

            model_name = metadata_dict.get("model_name") or args.lvlm_name
            if model_name not in allowed_lvlm_models:
                error_msg = f"Model not found or not enabled: {model_name}."
                print(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

            response = await model_manager.run(
                model_name,
                lambda model: model.generate(
                    metadata_dict["conv"],
                    images,
                    metadata_dict["max_new_tokens"],
                    metadata_dict["temperature"],
                    metadata_dict["top_p"],
                ),
            )

            return response

    if args.load_clip:
        @app.get("/clip_gen")
        async def clip_generation(request: Request):
            data = await request.json()
            text = data.get("text")

            response = await model_manager.run("clip", lambda model: model.encode(text))

            return response

    if args.load_sd:
        @app.get("/sd_gen")
        async def sd_generation(request: Request):
            data = await request.json()
            text = data.get("text")
            size_px = data.get("size_px")
            num_inference_steps = data.get("num_inference_steps")
            guidance_scale = data.get("guidance_scale")

            img = await model_manager.run(
                "stable_diffusion",
                lambda model: model.generate(text, size_px, num_inference_steps, guidance_scale),
            )
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            return StreamingResponse(buffer, media_type="image/png")

    if args.load_sem_relevance:
        @app.post("/sem_relevance")
        async def sem_relevance_generate(request: Request):
            # Get request data
            data = await request.json()
            text1 = data.get("text1", "")
            text2 = data.get("text2", "")
            similarity = await model_manager.run(
                "sem_relevance",
                lambda model: model.compute_similarity(text1, text2),
            )

            return {"similarity": similarity}

    if args.load_toxigen:
        @app.post("/toxigen")
        async def toxigen_generate(request: Request):
            # Get request data
            try:
                data = await request.json()
                texts = data.get("texts", [])
                texts_rebuild = []
                for text in texts:
                    if len(text) > 256:
                        text = text[:128]
                    texts_rebuild.append(text)
                result = await model_manager.run(
                    "toxigen",
                    lambda model: model.get_toxicity(texts_rebuild),
                )
            except Exception as e:
                # make sure the length of texts less than 512
                texts_rebuild = []
                for text in texts:
                    if len(text) > 256:
                        text = text[:256]
                    texts_rebuild.append(text)
                result = await model_manager.run(
                    "toxigen",
                    lambda model: model.get_toxicity(texts_rebuild),
                )
            return {"result": result}

    if args.load_llama_guard or args.load_toxigen:
        @app.post("/llama_guard")
        async def llama_guard_generate(files: List[UploadFile] = File(default=None), metadata: str = Form(...)):
            # Get request data
            metadata_dict = json.loads(metadata)
            images, error = await read_images(files)
            if error:
                return error

            response = await model_manager.run(
                "llama_guard",
                lambda model: model.generate(
                    metadata_dict["conv"],
                    images,
                    metadata_dict["max_new_tokens"],
                    metadata_dict["temperature"],
                    metadata_dict["top_p"],
                ),
            )

            return response

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    args = config().parse_args()

    main(args)
