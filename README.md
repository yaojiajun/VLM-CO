# VLMCoSolver: Vision-Language Models for Combinatorial Optimization

**"Vision-Language Models as End-to-End Combinatorial Optimization Solvers"**.

This repository provides a unified framework for solving combinatorial optimization problems using vision-language models (VLMs), specifically fine-tuned on **Qwen2.5-VL-7B-Instruct**.

![Framework](image/framework.png)

---

## 🎯 Supported Problems

We support 6 classic combinatorial optimization problems:

| Problem | Description | Instance Format |
|---------|-------------|-----------------|
| **TSP** | Traveling Salesman Problem | 2D node coordinates |
| **CVRP** | Capacitated Vehicle Routing Problem | 2D coordinates + demands + capacity |
| **MIS** | Maximum Independent Set | Graph adjacency |
| **MVC** | Minimum Vertex Cover | Graph adjacency |
| **JSSP** | Job Shop Scheduling Problem | Jobs × Machines matrix |
| **PFSP** | Permutation Flow Shop Problem | Jobs × Machines matrix |

---

## 🚀 Key Features

- **Unified Training Framework**: Single codebase for all 6 problems
- **Vision-Based Approach**: Converts problem instances to images for VLM processing
- **Two-Stage Training**:
  - **SFT (Supervised Fine-Tuning)**: Learn from expert solutions
  - **RL (Reinforcement Learning)**: GRPO-based policy optimization
- **Efficient Fine-Tuning**: LoRA with only 1.91% trainable parameters
- **Modular Design**: Problem-specific configurations via `prompts.py`

---

## 📁 Project Structure

```
VLMCoSolver/
├── sft_train_vision_unified.py    # Unified SFT training script
├── rl_train_vision_unified.py     # Unified RL training script (GRPO)
├── prompts.py                      # Problem-specific instruction builders
├── rewards.py                      # Reward functions for RL training
├── data/
│   ├── sft/                        # SFT training data
│   │   └── train/
│   │       ├── images/             # Problem instance visualizations
│   │       └── train_*.json        # Training annotations
│   └── rl/                         # RL training data
│       └── {problem}/
│           └── train/
│               ├── images/
│               └── train_rl.json
├── VisualFT/
│   └── cmd.sh                      # Model merging script (LoRA → full model)
└── VisualEvo/                      # Visual self-evolution design (VDEvo)
```

---

## 🛠️ Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 24GB+ GPU memory (recommended: A800, RX6000)

### Setup

```bash
# Clone the repository
git clone https://github.com/yaojiajun/VLM-CO.git
cd VLMCoSolver

# Install dependencies
pip install -r requirements.txt

# Download base model (Qwen2.5-VL-7B-Instruct)
# Place it in: ./models1_cache/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master
```

---

## 📊 Data Preparation

### Data Format

You can generate your own data through the problem-specific environments under `/Envs/`, or use the data generated in the original paper:

- **Text DATA**: https://drive.google.com/drive/folders/1bE1coGUa00gfuMkPXnfvldi1-WHGNnEb?usp=sharing
- **Image DATA**: https://drive.google.com/drive/folders/1VN9crftdW7DTsMQupbc06u6PzRT-Bwnx?usp=sharing
### Data Format

Each JSON record should contain:

```json
{
  "num_nodes": 20,              // For TSP/CVRP/MIS/MVC
  "vehicle_capacity": 50,       // For CVRP only
  "n_jobs": 10,                 // For JSSP/PFSP
  "n_machines": 5,              // For JSSP/PFSP
  "output": "Routes: [[0,1,2,0]], Objective: 15.3",
  "instance": [...]             // Raw instance data
}
```

### Image Naming Convention

- **TSP/CVRP/MIS/MVC**: `train_{idx:05d}_n{num_nodes}.png`
- **JSSP/PFSP**: `train_{idx:05d}_j{n_jobs}_m{n_machines}.png`

---

## 🎓 Training

### Stage 1: Supervised Fine-Tuning (SFT)

```bash
python sft_train_vision_unified.py \
    --problem_type cvrp \
    --model_name /path/to/Qwen2.5-VL-7B-Instruct \
    --images_dir ./data/sft/train/images \
    --output_dir ./output_cvrp_vision \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --max_seq_length 20000 \
    --disable_wandb
```

**Key Arguments**:
- `--problem_type`: Choose from `{tsp, cvrp, mis, mvc, jssp, pfsp}`
- `--disable_wandb`: Disable WandB logging (optional)
- `--load_in_4bit`: Enable 4-bit quantization for low memory

### Stage 2: Reinforcement Learning (GRPO)

```bash
python rl_train_vision_unified.py \
    --problem_type cvrp \
    --model_name ./output_cvrp_vision/checkpoint-5000 \
    --train_json ./data_rl/cvrp/train/train_rl.json \
    --images_dir ./data_rl/cvrp/train/images \
    --output_dir ./output_rl_cvrp \
    --num_generations 8 \
    --beta 0.05 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --disable_wandb
```

**GRPO Parameters**:
- `--num_generations`: Number of completions per prompt (G in GRPO)
- `--beta`: KL penalty coefficient
- `--max_new_tokens`: Maximum tokens for generation

---

## 🔧 Model Merging

After training, merge LoRA adapters into the base model:

```bash
cd VisualFT
./cmd.sh ../output_cvrp_vision/checkpoint-5000 ../merged_cvrp_model
```

This creates a standalone model without requiring LoRA at inference time.

---

## 📝 Prompt Design

All instruction prompts follow the specifications from our paper. See [`prompts.py`](prompts.py) for detailed implementations.

**Example (CVRP)**:
```
This is a CVRP instance with 1 depot and 19 customer nodes. Each blue circle 
represents a customer (labeled with node ID and demand in brackets), and the red 
star marks the depot (node 0). The vehicle capacity is 50 units. Each route must 
start and end at the depot, visit customers assigned to it exactly once, and ensure 
the total demand does not exceed capacity. The goal is to minimize total travel 
distance. Output format: Routes: [[0, ..., 0], [0, ..., 0]], Objective: <distance>
```

---

## 🎯 VLM-CO Reward Functions (RL)

Currently implemented:
- **CVRP**: Optimality reward + Feasibility penalty

To add reward functions for other problems, implement them in `rewards.py` following the CVRP template.

---

## 📈 Results

**

---

## 🙏 Acknowledgments

This work builds upon:
- [Unsloth](https://github.com/unslothai/unsloth) - Efficient LLM fine-tuning
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2-VL) - Vision-language foundation model
- [EOH](/https://arxiv.org/pdf/2401.02051) - Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model
- [Hercules](https://arxiv.org/pdf/2505.12627) - Efficient Heuristics Generation for Solving Combinatorial Optimization Problems Using Large Language Models
- [LLMCoSolver](https://github.com/Summer142857/LLMCoSolver) - Prior work on LLM-based CO solvers
