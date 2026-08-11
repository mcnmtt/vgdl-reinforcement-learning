# Come funziona la loss nel fine-tuning

## Meccanismo base

Il modello fa **next-token prediction**: dato un testo, deve predire il prossimo token. La loss misura quanto è sbagliato.

**Concretamente:** per ogni esempio del dataset, il testo completo è:
```
[system prompt] + [descrizione gioco] + [codice VGDL]
```
Il modello riceve tutto questo e deve predire token per token il codice VGDL alla fine. La **cross-entropy loss** confronta:
- cosa ha predetto il modello (distribuzione di probabilità sui token)
- cosa c'era scritto davvero nel codice VGDL di riferimento

Loss bassa = il modello assegna alta probabilità ai token giusti.

---

## Training loss vs Eval loss

| | Training loss | Eval loss |
|---|---|---|
| **Calcolata su** | `dataset["train"]` | `dataset["test"]` |
| **Quando** | durante ogni step di training (ogni 10 steps, `logging_steps=10`) | una volta per epoch, **senza aggiornare i pesi** |
| **A cosa serve** | vedere se il modello sta imparando | vedere se **generalizza** su dati mai visti |

---

## Cos'è `dataset["test"]`?

Viene da `load_from_disk("dataset_hf")`, che ha già una split `train`/`test` preesistente. Sono esempi di coppie (descrizione → VGDL) **mai usati nel training**. Il trainer li usa per calcolare la eval loss e per il `load_best_model_at_end=True` (salva il modello con la eval loss più bassa).

---

**In breve:** la loss confronta sempre il testo generato dal modello con il codice VGDL "corretto" del dataset. La differenza è solo su quali esempi viene calcolata.
