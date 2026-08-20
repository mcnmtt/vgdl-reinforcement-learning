# Phi-4-mini-instruct

Questa cartella contiene gli esperimenti per `microsoft/Phi-4-mini-instruct`.

La pipeline Phi completa è ora:

```text
zero-shot -> supervised learning (QLoRA) -> GRPO dall'adapter SFT
```

Il prompt condiviso tra SFT e GRPO è in `prompting.py`, così training e
inference non divergono accidentalmente.

Modello scelto per il confronto RL:

- famiglia diversa da Gemma e Qwen
- 3.8B parametri
- MIT license
- instruction-tuned
- gestibile in QLoRA meglio di un 7B, anche se GRPO resta pesante

## Ambiente consigliato

Usare l'ambiente gia' presente:

```powershell
conda activate rlia
```

Verifica rapida:

```powershell
python -c "import torch, transformers, trl, peft, bitsandbytes; print(torch.__version__); print(transformers.__version__)"
```

Se l'ambiente non contiene ancora le dipendenze specifiche della pipeline Phi:

```powershell
python -m pip install -r requirements.txt
```

## Download controllato

Scarica i pesi nella cache Hugging Face senza caricare il modello:

```powershell
python models/phi4-mini/zero-shot/inference/vgdl_gen_phi4mini_zeroshot.py --download-only
```

## Prima prova zero-shot batch

Lo script ha una configurazione in alto:

```python
DESCRIPTION_DIR = Path("dataset/descriptions")
OUTPUT_DIR = Path("models/phi4-mini/zero-shot/vgdl")
MAX_DESCRIPTIONS: int | None = 3
```

Con `MAX_DESCRIPTIONS = 3` genera i primi tre VGDL del dataset
(`1_artillery`, `2_lander`, `3_mario`). Con `MAX_DESCRIPTIONS = None`
processa tutta la cartella `dataset/descriptions`.

Comando standard:

```powershell
python models/phi4-mini/zero-shot/inference/vgdl_gen_phi4mini_zeroshot.py
```

Gli output vengono salvati in:

```text
models/phi4-mini/zero-shot/vgdl/
```

Per controllare prima quali file verranno processati e vedere il prompt:

```powershell
python models/phi4-mini/zero-shot/inference/vgdl_gen_phi4mini_zeroshot.py --dry-run
```

Puoi comunque sovrascrivere la variabile da terminale:

```powershell
python models/phi4-mini/zero-shot/inference/vgdl_gen_phi4mini_zeroshot.py --limit 3
python models/phi4-mini/zero-shot/inference/vgdl_gen_phi4mini_zeroshot.py --limit 0
```

Per generare un singolo file:

```powershell
python models/phi4-mini/zero-shot/inference/vgdl_gen_phi4mini_zeroshot.py --description-file dataset/descriptions/14_sokoban.txt --output-path models/phi4-mini/zero-shot/vgdl/14_sokoban_vgdl_phi4mini.txt
```

Lo script usa di default il supporto nativo `Phi3ForCausalLM` di Transformers.
Non usare `--trust-remote-code` salvo necessita' specifiche: nel nostro ambiente
il codice remoto del repository prova a importare `SlidingWindowCache`, non
presente in `transformers.cache_utils`.

## Supervised learning

Prima validare la preparazione del dataset, poi avviare il training:

```powershell
python models/phi4-mini/supervised-learning/finetune.py --dry-run
python models/phi4-mini/supervised-learning/finetune.py
```

L'adapter viene salvato in
`models/phi4-mini/supervised-learning/model-finetuned`. La loss è calcolata
soltanto sulla completion VGDL. I dettagli, le opzioni anti-OOM e i comandi di
inference sono in `supervised-learning/README.md`.

## Reinforcement learning dopo SFT

Il training GRPO richiede l'adapter supervisionato:

```powershell
python models/phi4-mini/reinforcement-learning/grpo_train.py
```

I nuovi risultati finiscono in
`models/phi4-mini/reinforcement-learning/grpo-output-sft`, separati dal vecchio
run `grpo-output` che non era partito dalla SFT.

Se compare CUDA out of memory:

- chiudere browser/app che usano GPU
- ridurre `--gpu-max-memory`, ad esempio `--gpu-max-memory 4GiB`
- provare prima solo `--download-only`
- per GRPO usare una macchina con almeno 12-16 GB VRAM

## Nota hardware

La macchina locale rilevata ha una RTX 2060 da 6 GB. L'inference 4-bit puo' essere tentata, ma il GRPO locale completo e' a rischio OOM. Per RL serio conviene usare una GPU da almeno 12 GB, meglio 16 GB.
