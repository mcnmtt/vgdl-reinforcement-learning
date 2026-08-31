# Generazione di VGDL con LLM e Reinforcement Learning

Progetto d'esame per il corso di **Intelligenza Artificiale** della Laurea
Magistrale in Informatica, track **Data Science e Machine Learning**,
dell'Universita' degli Studi di Salerno. Docente: **Vincenzo Deufemia**.

**Autori:** [Mattia Maucioni](https://github.com/mcnmtt) e [Antonio Landi](https://github.com/antonio-Landi).

## Obiettivo

Il progetto genera programmi in **Video Game Description Language (VGDL)** a
partire da descrizioni testuali di giochi. Confronta tre approcci:

- generazione zero-shot;
- fine-tuning supervisionato con QLoRA (SFT);
- reinforcement learning con Group Relative Policy Optimization (GRPO).

I modelli principali sono **Qwen3.5-4B**, **Gemma4-E4B-it** e
**Phi-4-mini-instruct**.

## Risultati

La valutazione usa 21 giochi e decodifica deterministica, con un massimo di
400 token generati. `Parser` indica gli output accettati da `py-vgdl`; non e'
una prova completa di esecuzione del gioco. La similarita' e' un indice Jaccard
pesato su sprite, interazioni e condizioni di terminazione.

| Modello | Condizione | Parser | Struttura completa | Similarita' | Tempo medio |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | Zero-shot | 100.0% | 0.0% | 0.0% | 25.3 s |
| Qwen3.5-4B | SFT | 28.6% | 42.9% | 20.2% | 36.0 s |
| Qwen3.5-4B | GRPO | 14.3% | 47.6% | 21.1% | 30.0 s |
| Gemma4-E4B-it | Zero-shot | 100.0% | 0.0% | 0.0% | 7.7 s |
| Gemma4-E4B-it | SFT | 23.8% | 33.3% | 18.4% | 55.8 s |
| Gemma4-E4B-it | GRPO | 14.3% | 23.8% | 14.9% | 50.7 s |
| Phi-4-mini | Zero-shot | 0.0% | 57.1% | 0.0% | 13.4 s |
| Phi-4-mini | SFT | 0.0% | 33.3% | 10.3% | 25.9 s |
| Phi-4-mini | GRPO | 0.0% | 23.8% | 9.4% | 27.8 s |

I dati usati per questa tabella sono in
[`evaluation/results/`](evaluation/results/), nelle cartelle `*-final`.

## Struttura

```text
dataset/        Specifiche VGDL, descrizioni testuali e script per il dataset
models/         Script zero-shot, SFT e GRPO per ogni modello
evaluation/     Metriche, valutatore e risultati delle valutazioni
py-vgdl/        Parser VGDL incluso nel progetto
```

## Avvio rapido

Sono necessari Python 3.10 e le dipendenze in
[`environment.yml`](environment.yml) e [`requirements.txt`](requirements.txt).

```bash
conda env create -f environment.yml
conda activate <nome-ambiente>
python -m pip install -e ./py-vgdl
python dataset/dataset_hf.py
```

I modelli Hugging Face possono richiedere l'accettazione della relativa licenza.
Per il caso zero-shot di Gemma4 servono anche Ollama e il modello locale:

```bash
ollama pull gemma4:e4b
```

Esempio di valutazione zero-shot di Gemma4:

```bash
python evaluation/evaluate_model_testset.py --backend ollama --model gemma4:e4b --run-name gemma4-zero-shot-final --max-new-tokens 400
```

Per addestrare un modello, eseguire lo script `finetune.py` oppure
`grpo_train.py` nella relativa cartella in `models/`. Per tutte le opzioni del
valutatore:

```bash
python evaluation/evaluate_model_testset.py --help
```
