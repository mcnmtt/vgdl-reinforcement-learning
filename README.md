# Generazione di VGDL con LLM e Reinforcement Learning

Progetto d'esame sulla generazione di programmi in **Video Game Description
Language (VGDL)** a partire da descrizioni di giochi in linguaggio naturale.
Il repository contiene l'intera pipeline sperimentale: costruzione del dataset,
baseline zero-shot, fine-tuning supervisionato con QLoRA, reinforcement learning
con GRPO e valutazione riproducibile.

I tre modelli selezionati per gli esperimenti principali sono:

- **Qwen3.5-4B**
- **Gemma4-E4B-it**
- **Phi-4-mini-instruct**

La relazione del progetto viene consegnata separatamente e non fa parte del
contenuto versionato della repository.

## Risultati principali

La valutazione hold-out corrente copre 21 giochi e usa il parser VGDL, insieme
a una similarita' strutturale Jaccard pesata su sprite, interazioni e condizioni
di terminazione.

| Condizione Gemma4 | VGDL eseguibile | Struttura completa | Similarita' strutturale | Tempo medio di generazione |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot, Ollama gemma4:e4b | 100.0% | 0.0% | 0.0% | 7.8 s |
| Fine-tuning supervisionato QLoRA | 0.0% | 85.7% | 26.1% | 107.2 s |

Gli output leggibili dalla macchina sono in
[evaluation/results/](evaluation/results); le figure e il report HTML sono in
[evaluation/plots/gemma4-comparison/](evaluation/plots/gemma4-comparison).
Il training GRPO impiega una reward che combina direttamente parsabilita',
similarita' strutturale, controlli sull'ontologia VGDL, gestione di EOS e
formattazione dell'output.

## Struttura della repository

```text
dataset/        File VGDL sorgenti, descrizioni in linguaggio naturale e builder del dataset
models/         Script di inferenza, SFT/QLoRA e GRPO per ciascun modello
evaluation/     Controlli di eseguibilita', metriche strutturali, risultati e grafici
py-vgdl/        Implementazione inclusa del parser VGDL
```

Dataset generati, pesi scaricati dei modelli, adapter LoRA, checkpoint, stati
dell'ottimizzatore e cache sono intenzionalmente esclusi da Git.

## Ambiente

Il progetto richiede Python 3.10 e le dipendenze dichiarate in
[environment.yml](environment.yml) e [requirements.txt](requirements.txt).

```bash
conda env create -f environment.yml
conda activate <nome-ambiente>
python -m pip install -e ./py-vgdl
```

Per il training su GPU, installare una build di PyTorch compatibile con CUDA e
con l'hardware disponibile. Le restanti dipendenze Python vengono installate
da [environment.yml](environment.yml).

I modelli Hugging Face possono richiedere autenticazione e accettazione delle
rispettive licenze. L'esperimento zero-shot di Gemma richiede inoltre
[Ollama](https://ollama.com/):

```bash
ollama pull gemma4:e4b
```

## Preparazione del dataset

Le descrizioni sorgenti e i file VGDL sono versionati. Prima di eseguire SFT,
GRPO o la valutazione sul test set, costruire il dataset locale Hugging Face:

```bash
python dataset/dataset_hf.py
```

`dataset/create_db.py` rigenera le descrizioni tramite API Anthropic.
E' opzionale e richiede `dataset/.env`, creato a partire da
`dataset/.env.example`.

## Workflow principali

Eseguire tutti i comandi dalla root della repository dopo aver attivato
l'ambiente Python configurato.

### Valutazione zero-shot con Gemma4

```bash
python evaluation/evaluate_model_testset.py --backend ollama --model gemma4:e4b --run-name gemma4-zero-shot
```

### Fine-tuning supervisionato

```bash
python models/gemma4/supervised-learning/finetune.py
python models/phi4-mini/supervised-learning/finetune.py
python models/qwen3.5/supervised-learning/finetune.py
```

Gli adapter sono salvati localmente nella directory `supervised-learning`
del rispettivo modello e sono ignorati da Git.

### Valutazione SFT

```bash
python evaluation/evaluate_model_testset.py --backend hf --model google/gemma-4-E4B-it --adapter models/gemma4/supervised-learning/model-finetuned --run-name gemma4-supervised --max-new-tokens 400
```

### Reinforcement learning con GRPO

```bash
python models/gemma4/reinforcement-learning/grpo_train.py
python models/phi4-mini/reinforcement-learning/grpo_train.py
python models/qwen3.5/reinforcement-learning/grpo_train.py
```

Il GRPO di Gemma4 e Phi-4-mini parte dal rispettivo adapter SFT. Gli script di
training mantengono localmente gli snapshot `best-model` e
`last-model` per la ripresa del training; tali snapshot non vengono
versionati.

### Grafici e report di valutazione

```bash
python evaluation/plot_evaluation_results.py --runs gemma4-zero-shot gemma4-supervised --output-dir evaluation/plots/gemma4-comparison
```

Per validare direttamente un singolo file VGDL:

```bash
python evaluation/check_vgdl_executability.py models/gemma4/zero-shot/vgdl/artillery_vgdl_gemma4.txt
```

## Politica di versionamento

La repository include codice sorgente, file sorgenti del dataset, esempi VGDL
generati selezionati, output delle valutazioni e grafici. Non include pesi dei
modelli, checkpoint, cache Hugging Face, credenziali API locali o i dati
rigenerati in `dataset_hf`. In questo modo il progetto consegnato resta
compatto, pur mantenendo i passaggi necessari a riprodurre ciascun esperimento.
