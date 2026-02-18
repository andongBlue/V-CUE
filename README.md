# V-CUE: Visual-Cultural Understanding Enhancement

Official implementation of **"Beyond Text: Enhancing Multicultural Robustness of LLMs via Generative Visual Cues"** (ACL 2025 Submission).

## Overview

V-CUE is a visual-space-assisted generation framework that enhances culturally sensitive tasks and mitigates cross-cultural bias in LLMs through generated visual cues.

**Core idea**: When an LLM detects uncertainty or potential cultural bias, V-CUE generates culturally relevant images based on the input text and feeds them back into the generation process.

### Architecture

```
Input Question
      │
      ▼
┌─────────────┐
│ Base LLM    │──► Initial Answer
└─────────────┘
      │
      ▼
┌─────────────┐
│ Uncertainty │──► True:  Return answer directly
│ Detection   │   False: Continue to VG
│ (UD)        │
└─────────────┘
      │ (uncertain)
      ▼
┌─────────────┐    c = [region] + [object] + [symbol]
│ Visual      │──► Extract cultural cues
│ Generation  │──► V = Text-to-Image(c)
│ (VG)        │
└─────────────┘
      │
      ▼
┌─────────────┐    LLM-text: F(C, V) = LLM(C ⊕ Cap(V))
│ Visual      │    LLM-VL:   F(C, V) = VLM(C, V)
│ Reconstruc. │──► Enhanced Answer
│ (VR)        │
└─────────────┘
```

## Project Structure

```
code/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml           # Default configuration
├── vcue/
│   ├── __init__.py
│   ├── pipeline.py            # Main V-CUE pipeline
│   ├── prompts.py             # All prompt templates (Appendix A-D)
│   ├── uncertainty_detection.py  # UD module
│   ├── visual_generation.py      # VG module (cue extraction + image gen)
│   ├── visual_reconstruction.py  # VR module (text/VL pathways)
│   └── models/
│       ├── __init__.py
│       ├── llm_client.py      # Unified LLM API client
│       ├── image_generator.py # Stable Diffusion wrapper
│       └── image_captioner.py # Image-to-text captioner
├── evaluation/
│   ├── __init__.py
│   ├── culturalbench_eval.py  # CulturalBench evaluation
│   ├── care_eval.py           # CARE-eval evaluation
│   └── metrics.py             # Accuracy, correction rate, CARE judge
├── scripts/
│   ├── run_culturalbench.py   # Run CulturalBench experiments
│   └── run_care.py            # Run CARE experiments
└── data/
    └── .gitkeep
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.10
- NVIDIA GPU with >= 24GB VRAM (for Stable Diffusion)
- API keys for LLM services

## Configuration

Edit `config/default.yaml` or create a custom config file:

```yaml
# Set your API keys
llm:
  provider: "deepseek"
  model_name: "deepseek-chat"
  api_key: "your-deepseek-key"

# Or use environment variables
# export OPENAI_API_KEY=sk-xxx
# export DEEPSEEK_API_KEY=xxx
# export DASHSCOPE_API_KEY=xxx
```

### Supported Models

| System Type | Models |
|---|---|
| LLM-text | DeepSeek-V3, Qwen3-235B-A22B, Qwen3-32B, GPT-3.5-turbo |
| LLM-reasoning | Qwen3-235B-A22B, Qwen3-32B, DeepSeek-R1, o3-mini |
| LLM-VL | Qwen2.5-VL-72B, GPT-4o |

## Quick Start

### CulturalBench Evaluation

```bash
# Full evaluation (Multiple-Choice + True/False, 5 runs)
python scripts/run_culturalbench.py --config config/default.yaml

# Single task with specific model
python scripts/run_culturalbench.py --system_type text --model deepseek-chat --task mc

# Quick test with limited samples
python scripts/run_culturalbench.py --max_samples 50 --num_runs 1
```

### CARE-eval Evaluation

```bash
# Full CARE evaluation (Chinese, Arabic, Japanese)
python scripts/run_care.py --config config/default.yaml

# Specific cultures
python scripts/run_care.py --cultures chinese japanese --num_runs 3
```

## Benchmarks

| Benchmark | Task | Metrics |
|---|---|---|
| [CulturalBench](https://huggingface.co/datasets/kellycyy/CulturalBench) | Multiple-Choice, True/False | Accuracy |
| [CARE-eval](https://huggingface.co/datasets/geyang627/CARE-eval) | Cultural Generation | Generality, Cultural Relevance, Literary Quality (1-10) |

## Key Results

### CulturalBench (Table 1)

V-CUE consistently improves accuracy across all model types:
- **LLM-text**: +3.1 (MC) / +6.8 (TF) on average
- **LLM-reasoning**: up to +19.06 (MC) / +7.38 (TF)
- **LLM-VL**: +2.60 (MC) / +3.28 (TF)

### CARE-eval (Figure 3)

V-CUE yields consistent gains across Chinese, Arabic, and Japanese cultures, with the most significant improvements on Chinese-based LLMs.

## Image Generation Settings

Fixed across all experiments (Appendix F):
- Model: Stable Diffusion v2.1
- Guidance scale: 7.5
- Denoising steps: 30
- Resolution: 512×512
- Single fixed random seed per sample
- GPU: NVIDIA A100 (80GB)

## Citation

```bibtex
@inproceedings{vcue2025,
  title={Beyond Text: Enhancing Multicultural Robustness of LLMs via Generative Visual Cues},
  author={Anonymous},
  booktitle={ACL},
  year={2025}
}
```

## License

This project is released under the MIT License.
