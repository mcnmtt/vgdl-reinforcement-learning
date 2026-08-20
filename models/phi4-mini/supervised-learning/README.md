# Phi-4-mini Supervised Learning

Questa cartella contiene la fase SFT di `microsoft/Phi-4-mini-instruct` per la
generazione description-to-VGDL.

La pipeline usa:

- QLoRA 4-bit NF4 con calcolo `float16`;
- LoRA `r=16`, `alpha=32`, `dropout=0.05` sui moduli Phi
  `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`;
- esempi TRL nel formato `prompt`/`completion`;
- `completion_only_loss=True`, quindi i token del prompt non contribuiscono
  alla loss;
- contesto predefinito da 2048 token;
- esclusione delle coppie vuote e, per impostazione predefinita, degli esempi
  oltre il limite, così il modello non impara VGDL troncati.

Il prompt è definito una sola volta in `models/phi4-mini/prompting.py` ed è
condiviso con il training GRPO e le rispettive inferenze.

## Controllo preliminare

Eseguire dalla root del repository:

```powershell
conda activate rlia
python -m pip install -r requirements.txt
python models/phi4-mini/supervised-learning/finetune.py --dry-run
```

Sul dataset corrente, con `--max-length 2048`, il controllo seleziona 169
esempi di train e 20 di test. Scarta una coppia senza codice VGDL e 11 coppie
oltre il limite complessivo. Per mantenere anche queste ultime, usare
`--keep-overlength`; TRL dovrà però troncarle.

## Training

```powershell
conda activate rlia
python models/phi4-mini/supervised-learning/finetune.py
```

L'adapter finale e le metriche vengono salvati in:

```text
models/phi4-mini/supervised-learning/model-finetuned/
```

Il resume dall'ultimo `checkpoint-*` è automatico. Per un nuovo esperimento
usare un'altra cartella:

```powershell
python models/phi4-mini/supervised-learning/finetune.py `
  --output-dir models/phi4-mini/supervised-learning/runs/experiment-02
```

In caso di CUDA OOM sulla RTX 2060 da 6 GB, il primo compromesso consigliato è:

```powershell
python models/phi4-mini/supervised-learning/finetune.py --max-length 1536
```

Con 1536 token vengono esclusi più esempi lunghi; il dry-run mostra sempre il
conteggio effettivo prima del training.

## Inference SFT

Per un singolo gioco:

```powershell
python models/phi4-mini/supervised-learning/inference.py `
  --description-file dataset/descriptions/14_sokoban.txt
```

Per i primi tre file, oppure per l'intero dataset:

```powershell
python models/phi4-mini/supervised-learning/inference.py --limit 3
python models/phi4-mini/supervised-learning/inference.py --limit 0
```

Gli output batch vengono scritti in
`models/phi4-mini/supervised-learning/vgdl/`. Per confrontare adapter e modello
base sullo stesso prompt si può aggiungere `--compare-base`.

## Passaggio a GRPO

`grpo_train.py` richiede questo adapter SFT e non crea più automaticamente una
LoRA nuova. Dopo la SFT:

```powershell
python models/phi4-mini/reinforcement-learning/grpo_train.py
```

Il nuovo esperimento GRPO viene salvato in `grpo-output-sft`, separatamente dal
precedente `grpo-output`, che era partito direttamente dal modello base.
