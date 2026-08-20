# VGDL Generation with LLMs and Reinforcement Learning

Course project on generating **Video Game Description Language (VGDL)** programs
from natural-language game descriptions. The repository contains the full
experimental pipeline: dataset construction, zero-shot baselines, supervised
fine-tuning with QLoRA, GRPO reinforcement learning, and reproducible
evaluation.

The three models selected for the main experiments are:

- **Qwen3.5-4B**
- **Gemma4-E4B-it**
- **Phi-4-mini-instruct**

The accompanying report is available as
[PDF](docs/vgdl-reinforcement-learning-report.pdf); its LaTeX source is in
[docs/](docs).

## Results at a glance

The held-out evaluation currently covers 21 games and uses the VGDL parser plus
a weighted structural Jaccard similarity over sprites, interactions, and
termination conditions.

| Gemma4 condition | Executable VGDL | Complete structure | Structural similarity | Mean generation time |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot, Ollama gemma4:e4b | 100.0% | 0.0% | 0.0% | 7.8 s |
| QLoRA supervised fine-tuning | 0.0% | 85.7% | 26.1% | 107.2 s |

The corresponding machine-readable outputs are tracked in
[evaluation/results/](evaluation/results), while figures and the HTML report
are in [evaluation/plots/gemma4-comparison/](evaluation/plots/gemma4-comparison).
GRPO training uses a reward that directly combines parsability, structural
similarity, VGDL ontology checks, EOS handling, and output formatting.

## Repository layout

```text
dataset/        Source VGDL files, natural-language descriptions, and dataset builder
models/         Inference, SFT/QLoRA, and GRPO scripts for each model
evaluation/     Executability checks, structural metrics, result files, and plots
py-vgdl/        Local VGDL parser dependency
docs/           Exam report (LaTeX source and compiled PDF)
```

Generated datasets, downloaded model weights, LoRA adapters, checkpoints,
optimizer states, and caches are intentionally excluded from Git.

## Environment

The project was developed with the Conda environment **rlia** on Windows and an
NVIDIA GPU. Python 3.10 is the supported baseline.

```powershell
conda env create -f environment.yml
conda activate rlia
python -m pip install -e .\py-vgdl
```

For GPU training, install a CUDA-compatible PyTorch build for the local driver
before running the training scripts. The remaining Python dependencies are
declared in requirements.txt and are installed by environment.yml.

The Hugging Face models may require authentication and acceptance of their
respective model licenses. The zero-shot Gemma experiment additionally requires
[Ollama](https://ollama.com/):

```powershell
ollama pull gemma4:e4b
```

## Dataset preparation

The source descriptions and VGDL files are versioned. Build the local Hugging
Face dataset before SFT, GRPO, or test-set evaluation:

```powershell
python dataset/dataset_hf.py
```

dataset/create_db.py regenerates descriptions through the Anthropic API. It is
optional and requires dataset/.env, created from dataset/.env.example.

## Main workflows

Run all commands from the repository root after activating rlia.

### Zero-shot evaluation with Gemma4

```powershell
python evaluation/evaluate_model_testset.py --backend ollama --model gemma4:e4b --run-name gemma4-zero-shot
```

### Supervised fine-tuning

```powershell
python models/gemma4/supervised-learning/finetune.py
python models/phi4-mini/supervised-learning/finetune.py
python models/qwen3.5/supervised-learning/finetune.py
```

Adapters are saved locally below each model's supervised-learning directory and
are ignored by Git.

### SFT evaluation

```powershell
python evaluation/evaluate_model_testset.py --backend hf --model google/gemma-4-E4B-it --adapter models/gemma4/supervised-learning/model-finetuned --run-name gemma4-supervised --max-new-tokens 400
```

### GRPO reinforcement learning

```powershell
python models/gemma4/reinforcement-learning/grpo_train.py
python models/phi4-mini/reinforcement-learning/grpo_train.py
python models/qwen3.5/reinforcement-learning/grpo_train.py
```

Gemma4 and Phi-4-mini GRPO start from their SFT adapter. The training scripts
preserve local best-model and last-model snapshots for resuming; these snapshots
are deliberately not committed.

### Plots and reports

```powershell
python evaluation/plot_evaluation_results.py --runs gemma4-zero-shot gemma4-supervised --output-dir evaluation/plots/gemma4-comparison
```

To validate one VGDL file directly:

```powershell
python evaluation/check_vgdl_executability.py models/gemma4/zero-shot/vgdl/artillery_vgdl_gemma4.txt
```

## Reproducing the report

The report is self-contained: its charts are generated directly with TikZ/PGF,
so no raster training plots are required.

```powershell
cd docs
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Version-control policy

The repository tracks source code, dataset source files, selected generated
VGDL examples, evaluation outputs, plots, and documentation. It does **not**
track model weights, checkpoints, Hugging Face caches, local API credentials,
or regenerated dataset_hf data. This keeps the submitted project compact while
preserving the steps needed to reproduce each experiment.
