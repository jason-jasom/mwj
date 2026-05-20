# MAPA: Multi-Turn Adaptive Prompting Attack on Large Vision-Language Models

## Compute Resource

### Hardware Requirements
- **GPUs**: 2× NVIDIA H100 80GB GPUs (160GB total GPU memory)

Single experiment setup loads target model, CLIP model, and judge models across 2 GPUs using `device_map="auto"` for optimal memory distribution.

## Setup

### Model Download
Download the required models from Hugging Face and save them under the `models/` folder:

```bash
# Create models directory
mkdir -p models

# Download target LVLMs
git clone https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf models/llava-hf_llava-v1.6-mistral-7b-hf
git clone https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct models/Qwen_Qwen2-VL-7B-Instruct  
git clone https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct models/meta-llama_Llama-3.2-11B-Vision-Instruct

# Download judge models
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct models/meta-llama_Llama-3.1-8B-Instruct
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 models/mistralai_Mistral-7B-Instruct-v0.3
git clone https://huggingface.co/mistralai/Mistral-Small-Instruct-2409 models/mistralai_Mistral-Small-Instruct-2409

# Download CLIP and other auxiliary models (auto-downloaded during first run)
```

### Environment Setup
```bash
# Create and activate conda environment
conda create -n mwj python=3.10
conda activate mwj

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Navigate to attack directory and install dependencies
cd attack
pip install -r requirements.txt
```

## How to run the project

### 1. Start Model Server
```bash
# Example: Start LLaVA server
python server.py --lvlm_name=llava --load_clip --load_sem_relevance --port=8000
```
Available target models: `llava`, `qwen`, `llama`

### 2. Run Attack
```bash
# Example: Run MAPA attack on HarmBench
python main.py --dataset=harmbench --task_i_start_from=0 --num_tasks=60 --experiment_name=llava_test --port=8000
```
Available datasets: `harmbench`, `harmbench_tiny`, `jailbreakbench`