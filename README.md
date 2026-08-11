# VGDL Reinforcement Learning

Repository per il progetto d'esame sulla generazione automatica di codice VGDL a partire da descrizioni in linguaggio naturale tramite Large Language Model.

Il progetto segue una pipeline in tre fasi:

1. costruzione del dataset VGDL/descrizione;
2. valutazione zero-shot di diversi modelli;
3. selezione dei modelli migliori e confronto tra supervised fine-tuning e reinforcement learning.

## Pipeline

```text
dataset/vgdl_files/        -> specifiche VGDL originali
dataset/create_db.py       -> genera descrizioni testuali tramite API Anthropic
dataset/descriptions/      -> descrizioni in linguaggio naturale
dataset/dataset_hf.py      -> converte i dati in formato Hugging Face Dataset

models/<model>/zero-shot/              -> inferenza zero-shot
models/<model>/supervised-learning/    -> QLoRA/SFT
models/<model>/reinforcement-learning/ -> GRPO/RL

evaluation/                 -> validazione, similarity, metriche e plot
latex/                      -> mini-paper del progetto
```

## Modelli

La fase zero-shot e' stata usata come baseline esplorativa su piu' modelli open-weight. Dopo questa fase sono stati selezionati tre modelli principali per SFT e RL:

- Qwen3.5
- DeepSeek
- Phi-3.5

Gemma4 e' presente nella repository come esperimento aggiuntivo/di supporto, con script zero-shot, supervised learning e reinforcement learning.

## Setup

L'ambiente principale usato per training, inferenza e valutazione e':

```powershell
conda activate rlia
```

Le dipendenze principali sono raccolte in:

- `requirements.txt`
- `environment.yml`

Per la generazione delle descrizioni e' necessario creare localmente:

```text
dataset/.env
```

partendo da:

```text
dataset/.env.example
```

Il file `.env` non va versionato.

## Dataset

Per rigenerare il dataset:

```powershell
conda activate rlia
cd C:\Users\Mattia\Desktop\REPOs\vgdl-reinforcement-learning
python dataset/create_db.py
python dataset/dataset_hf.py
```

La cartella `dataset_hf/` e gli archivi compressi sono esclusi dal versionamento per evitare artefatti pesanti o rigenerabili.

## Zero-Shot

Ogni modello ha uno script dedicato nella rispettiva cartella:

```powershell
python models/qwen3.5/zero-shot/inference/vgdl_gen_qwen_zeroshot.py
python models/deepseek/zero-shot/inference/vgdl_gen_deepseek_zeroshot.py
python models/gemma4/zero-shot/inference/vgdl_gen_gemma4_zeroshot.py
```

## Supervised Fine-Tuning

Gli script SFT usano LoRA/QLoRA con configurazioni compatibili con una GPU consumer.

```powershell
python models/qwen3.5/supervised-learning/finetune.py
python models/deepseek/supervised-learning/finetune.py
python models/gemma4/supervised-learning/finetune.py
```

I checkpoint e gli adapter generati sono esclusi dal commit. Nel repository restano gli script necessari a riprodurre il training.

## Reinforcement Learning

La fase RL usa GRPO con reward composita basata su:

- eseguibilita' tramite parser `py-vgdl`;
- similarita' strutturale con il VGDL target;
- presenza dei blocchi fondamentali VGDL;
- validita' di sprite class, interaction functions e termination conditions;
- rispetto del formato raw VGDL senza markdown.

Esempio:

```powershell
python models/gemma4/reinforcement-learning/grpo_train.py
```

I modelli salvano checkpoint locali, `last-model` e `best-model`, ma questi artefatti non vengono versionati.

## Evaluation

Per valutare un modello sul test set:

```powershell
python evaluation/evaluate_model_testset.py `
  --backend ollama `
  --model gemma4:e4b `
  --run-name gemma4-zero-shot
```

```powershell
python evaluation/evaluate_model_testset.py `
  --backend hf `
  --model google/gemma-4-E4B-it `
  --adapter models/gemma4/supervised-learning/model-finetuned `
  --run-name gemma4-supervised `
  --max-new-tokens 400
```

Per creare plot comparativi:

```powershell
python evaluation/plot_evaluation_results.py `
  --runs gemma4-zero-shot gemma4-supervised `
  --output-dir evaluation/plots/gemma4-comparison
```

## Mini-Paper

Il mini-paper si trova in:

```text
latex/main.tex
```

Contiene descrizione del dataset, selezione dei modelli, protocollo sperimentale, SFT, RL e discussione dei risultati. La sezione Phi-3.5 contiene placeholder da completare quando saranno disponibili i risultati finali.

## Artefatti Esclusi Dal Versionamento

Sono esclusi dal commit:

- file `.env`;
- checkpoint di training;
- adapter `.safetensors`;
- optimizer/scheduler state;
- cache Unsloth;
- dataset Hugging Face rigenerabile;
- archivi `.zip`;
- cartelle `model-finetuned/` e `grpo-output/`.

Questa scelta mantiene la repository leggera e riproducibile.
