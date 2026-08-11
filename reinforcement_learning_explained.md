# Reinforcement Learning per la Generazione di VGDL

Questo documento spiega nel dettaglio come funziona il training con Reinforcement Learning applicato alla generazione di codice VGDL da descrizioni testuali. Copre l'algoritmo GRPO, il sistema di reward functions, i parametri di training e il flusso completo end-to-end.

---

## Indice

1. [Contesto: perché RL dopo SFT?](#1-contesto)
2. [GRPO: l'algoritmo](#2-grpo)
3. [Il sistema di reward functions](#3-reward-functions)
4. [Architettura del sistema di reward](#4-architettura-reward)
5. [Hyperparametri e loro ruolo](#5-hyperparametri)
6. [Flusso di training completo](#6-flusso-di-training)
7. [Come interpretare le metriche](#7-metriche)

---

## 1. Contesto

### Perché RL dopo SFT?

Il training avviene in due fasi sequenziali:

```
Qwen3.5-4B (base)
       │
       ▼
  SFT (Supervised Fine-Tuning)
  ─────────────────────────────
  Il modello impara a imitare
  il VGDL del dataset usando
  cross-entropy loss sul prossimo
  token. Ottimizza per "assomigliare"
  ai dati di training, non per
  "generare VGDL valido".
       │
       ▼
  GRPO (Reinforcement Learning)
  ─────────────────────────────
  Il modello impara a massimizzare
  reward esplicite: eseguibilita',
  correttezza semantica, struttura.
  Ottimizza per "qualita' reale",
  non per imitazione.
```

**Il problema del SFT da solo:** il modello impara a predire i token del training set, ma non ha feedback diretto su se il VGDL generato sia eseguibile o semanticamente corretto. Puo' imparare pattern superficiali che sembrano VGDL ma falliscono al parsing.

**Il vantaggio dell'RL:** il modello riceve un segnale esplicito per ogni output generato. Se il VGDL e' valido ottiene reward alta; se contiene errori ottiene reward bassa. Questo spinge il modello oltre la semplice imitazione.

---

## 2. GRPO

### Cos'è GRPO

**GRPO** (Group Relative Policy Optimization) è un algoritmo di RL sviluppato da DeepSeek per il training di LLM. È una variante di PPO semplificata, ottimizzata per non richiedere un modello di valore (value model) separato.

Riferimento: DeepSeek-R1 (2024).

### Intuizione di base

L'idea centrale è **la reward relativa all'interno di un gruppo**:

```
Per ogni prompt, genera G completions diverse.
Valuta ciascuna con le reward functions.
Non usare il valore assoluto della reward,
ma il rank relativo all'interno del gruppo.
```

Questo risolve due problemi:
- La scala assoluta della reward non importa (5.0 è "buona" solo se le altre sono < 5.0)
- Non serve calibrare le reward functions in modo assoluto

### Matematica di GRPO

Per ogni prompt `q`, GRPO genera `G` output `{o_1, ..., o_G}` e calcola i reward `{r_1, ..., r_G}`.

**Advantage normalizzato:**

```
A_i = (r_i - mean(r_1,...,r_G)) / std(r_1,...,r_G)
```

Dove `mean` e `std` sono calcolati sul gruppo di G completions.

**Obiettivo di training (GRPO loss):**

```
L_GRPO = -E[ A_i * log π_θ(o_i | q) ] + β * KL(π_θ || π_ref)
```

- `π_θ`: policy corrente (modello che stiamo allenando)
- `π_ref`: policy di riferimento (SFT model, frozen)
- `β`: coefficiente di penalità KL

Il termine KL impedisce al modello di degenerare verso output che massimizzano le reward senza senso (reward hacking).

### Perché non PPO?

PPO richiede un **critic** (modello separato che stima il valore atteso di ogni stato). Per un LLM, questo significa un secondo modello grande quanto quello principale. GRPO elimina questa necessità usando la normalizzazione di gruppo come baseline implicita.

```
PPO:  reward(o) - V(o)      ← V(o) è il critic, serve un secondo modello
GRPO: reward(o) - mean(rewards nel gruppo)  ← baseline calcolata on-the-fly
```

### Il ruolo di vLLM

Il token di generazione durante GRPO è lento: per ogni batch, il trainer genera `G * batch_size` completions. vLLM accelera questa fase con batching ottimizzato e memory management avanzato, riducendo il tempo di generazione di 3-5x rispetto alla generazione standard di HuggingFace.

---

## 3. Reward Functions

Il sistema di reward è composto da **8 funzioni indipendenti** con pesi diversi. GRPOTrainer somma i valori restituiti da tutte le funzioni per ottenere la reward totale.

### Panoramica dei pesi

| Funzione | Range | Peso massimo | Obiettivo |
|---|---|---|---|
| `reward_executability_shaped` | [0.0, 3.0] | **3.0** | VGDL eseguibile |
| `reward_similarity` | [0.0, 2.0] | **2.0** | Corretto rispetto al ref |
| `reward_structure` | [0.0, 1.5] | **1.5** | Struttura completa |
| `reward_valid_sprite_classes` | [0.5, 1.0] | **1.0** | Classi sprite valide |
| `reward_valid_interactions` | [0.5, 1.0] | **1.0** | Effetti interazione validi |
| `reward_valid_terminations` | [0.25, 0.5] | **0.5** | Terminazioni valide |
| `reward_eos_boundary` | [0.0, 0.5] | **0.5** | Uso corretto di EOS |
| `reward_no_markdown` | [-0.5, 0.0] | **0.0** | Formato puro |
| **Totale** | [-0.5, 9.5] | **9.5** | |

---

### 3.1 `reward_executability_shaped` (peso ×3.0)

**Motivazione:** la reward più importante. Se il VGDL non è eseguibile, il gioco non funziona. Era originariamente binaria (0 o 1), il che causava **sparse reward**: all'inizio del training quasi tutti gli output falliscono, il segnale di gradiente è quasi nullo.

**Soluzione: credito parziale (reward shaping)**

```
VGDL completamente valido → 3.0

Struttura parziale (non valido):
  +0.1  se inizia con "BasicGame"
  +0.1  per SpriteSet presente
  +0.1  per LevelMapping presente
  +0.1  per InteractionSet presente
  +0.1  per TerminationSet presente
  +0.3  se il parsing non fallisce subito
         (errori semantici invece di sintattici
          → la struttura è corretta, mancano definizioni)

Max parziale: 0.95 × 3.0 = 2.85  (mai uguale al caso valido)
```

**Validazione usata (da `_validate_vgdl_string`):**
1. Parsing completo via `VGDLParser` di py-vgdl
2. Controllo che tutti gli sprite nelle interazioni siano definiti nella SpriteSet
3. Controllo che gli sprite nelle terminazioni esistano

---

### 3.2 `reward_similarity` (peso ×2.0)

**Motivazione:** le reward sintattiche non garantiscono che il VGDL generato rappresenti il gioco descritto. Un VGDL può essere perfettamente valido ma descrivere un gioco completamente diverso da quello nella descrizione.

Questa reward misura quanto il VGDL generato **corrisponde semanticamente** al VGDL di riferimento del dataset usando la **similarità di Jaccard** su tre componenti.

**Formula:**

```
similarity = 0.25 × sprite_sim + 0.50 × interaction_sim + 0.25 × term_sim
```

Dove `jaccard(A, B) = |A ∩ B| / |A ∪ B|`.

**Componente Sprite (25%):**
- Estrae i nomi dei leaf sprite dalla SpriteSet (esclude sprite astratti come `structure`, `moving`)
- Jaccard tra il set del generato e il set del riferimento

**Componente Interazioni (50% — più importante):**
- Estrae tuple `(sprite1, sprite2, effetto)` dall'InteractionSet
- Normalizza i nomi degli effetti sinonimi (es. `stepBack` → `block`, `wallStop` → `block`)
- Tratta alcune interazioni come simmetriche (es. `kill`)
- Jaccard tra i set di tuple

**Componente Terminazioni (25%):**
- Estrae tuple `(stype, win)` dal TerminationSet
- Jaccard tra i set di condizioni

**Accesso al riferimento:** la colonna `vgdl` del dataset viene passata come kwarg da GRPOTrainer alle reward functions (per questo il dataset non rimuove la colonna `vgdl`).

---

### 3.3 `reward_structure` (peso ×1.5)

**Motivazione:** garantire che il VGDL abbia la struttura di primo livello corretta. Complementare a `reward_executability_shaped`, ma più veloce da calcolare (solo regex, nessun parsing).

```
Score base-1 (poi × 1.5):
  +0.2  se il testo inizia con "BasicGame"
  +0.2  per SpriteSet presente
  +0.2  per LevelMapping presente
  +0.2  per InteractionSet presente
  +0.2  per TerminationSet presente
  ─────
  Max: 1.0 × 1.5 = 1.5
```

---

### 3.4 `reward_valid_sprite_classes` (peso ×1.0)

**Motivazione:** VGDL ha un'ontologia fissa di classi sprite. Usare classi inventate (es. `Enemy`, `Player`) fa fallire il parser. Questa reward spinge il modello a usare solo le 33 classi valide.

**Come funziona:**
- Cerca tutti i pattern `> ClassName` nel testo (sintassi VGDL per assegnare una classe)
- Conta quante sono in `VALID_SPRITE_CLASSES` (33 classi dall'ontologia py-vgdl)
- Score = `valid_count / total_count`
- Score neutro **0.5** se non trova classi (non si penalizza output incompleto)

**Classi valide per categoria:**

| Categoria | Classi |
|---|---|
| Statici | `Immovable`, `Passive`, `ResourcePack`, `Flicker`, `Spreader` |
| Orientati | `Conveyor`, `Missile`, `OrientedFlicker`, `Walker`, `WalkJumper` |
| NPC | `RandomNPC`, `RandomInertial`, `RandomMissile`, `ErraticMissile`, `Bomber`, `Chaser`, `Fleeing`, `AStarChaser` |
| Producer | `SpawnPoint`, `Portal` |
| Avatar | `MovingAvatar`, `HorizontalAvatar`, `VerticalAvatar`, `FlakAvatar`, `OrientedAvatar`, `RotatingAvatar`, `RotatingFlippingAvatar`, `NoisyRotatingFlippingAvatar`, `ShootAvatar`, `AimedAvatar`, `AimedFlakAvatar`, `InertialAvatar`, `MarioAvatar` |

---

### 3.5 `reward_valid_interactions` (peso ×1.0)

**Motivazione:** analogamente alle classi sprite, gli effetti di interazione devono essere dall'ontologia py-vgdl.

**Come funziona:**
- Isola la sezione `InteractionSet` con regex
- Cerca tutti i pattern `> effectName` (minuscola iniziale)
- Score = `valid_count / total_count`
- Score neutro **0.5** se la sezione non è trovata

**Effetti validi (30 totali):**

`killSprite`, `cloneSprite`, `transformTo`, `stepBack`, `undoAll`, `bounceForward`, `bounceDirection`, `wallBounce`, `wallStop`, `conveySprite`, `windGust`, `slipForward`, `pullWithIt`, `attractGaze`, `turnAround`, `reverseDirection`, `flipDirection`, `wrapAround`, `teleportToExit`, `killIfSlow`, `killIfFromAbove`, `killIfAlive`, `killIfHasMore`, `killIfHasLess`, `killIfOtherHasMore`, `killIfOtherHasLess`, `collectResource`, `changeResource`, `spawnIfHasMore`

---

### 3.6 `reward_valid_terminations` (peso ×0.5)

**Motivazione:** le condizioni di terminazione devono essere una di tre keyword valide. Peso ridotto perché questa reward è già parzialmente coperta da `reward_executability_shaped`.

**Condizioni valide:** `Timeout`, `SpriteCounter`, `MultiSpriteCounter`

Score = `valid_count / total_count` (0.5 neutro se la sezione non è trovata).

---

### 3.7 `reward_eos_boundary` (peso ×0.5)

**Motivazione:** in VGDL il bordo dello schermo si chiama `EOS` (End Of Screen). I modelli tendono a inventare nomi come `edge`, `screen`, `wall_bound`, `boundary` che sono invalidi.

```
  1.0 × 0.5 = 0.5   se usa EOS
  0.0              se usa keyword non valide
  0.5 × 0.5 = 0.25  neutro (nessuna menzione del bordo)
```

---

### 3.8 `reward_no_markdown` (penalità, range [-0.5, 0.0])

**Motivazione:** il prompt di sistema chiede esplicitamente VGDL puro senza markdown o spiegazioni, ma i modelli tendono ad aggiungere fences ` ``` ` o frasi introduttive ("Here is the VGDL code..."). Questa penalità è pura deterrenza.

```
-0.5  se contiene backtick ``` (markdown code block)
-0.2  se inizia con prosa introduttiva
       (Here, This, The game, I have, Sure, Below, Let me, I'll)
 0.0  altrimenti
```

---

## 4. Architettura del sistema di reward

### Come interagiscono le reward

```
Per ogni prompt nel batch:
  ┌─────────────────────────────────────┐
  │  Genera 8 completions (G=8)         │
  │  con temperature=0.8                │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │  Per ogni completion c_i:           │
  │                                     │
  │  r_exec  = exec_shaped(c_i)  × 3.0 │
  │  r_sim   = similarity(c_i)   × 2.0 │
  │  r_struc = structure(c_i)    × 1.5 │
  │  r_spr   = sprite_cls(c_i)   × 1.0 │
  │  r_int   = interactions(c_i) × 1.0 │
  │  r_term  = terminations(c_i) × 0.5 │
  │  r_eos   = eos_boundary(c_i) × 0.5 │
  │  r_md    = no_markdown(c_i)        │
  │                                     │
  │  R_i = r_exec + r_sim + r_struc +  │
  │        r_spr + r_int + r_term +    │
  │        r_eos + r_md                │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │  Normalizzazione di gruppo:         │
  │                                     │
  │  μ = mean(R_1,...,R_8)             │
  │  σ = std(R_1,...,R_8)              │
  │  A_i = (R_i - μ) / σ              │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │  GRPO loss:                         │
  │  L = -E[A_i * log π(c_i | prompt)] │
  │    + β * KL(π || π_ref)            │
  │                                     │
  │  Update dei parametri LoRA          │
  └─────────────────────────────────────┘
```

### Perché questi pesi

Il principio di pesatura riflette la gerarchia di importanza:

1. **Eseguibilità (×3):** un VGDL non eseguibile è inutilizzabile, indipendentemente da tutto il resto. È il vincolo più stretto.

2. **Similarità (×2):** un VGDL eseguibile che non corrisponde alla descrizione è ugualmente inutile per il task. Questa reward è l'unica che misura la correttezza *semantica* rispetto all'input.

3. **Struttura (×1.5):** necessaria ma già parzialmente coperta dall'eseguibilità. Il peso è intermedio perché fornisce segnale utile anche quando il parser fallisce.

4. **Ontologia sprite/interazioni (×1):** dettaglio tecnico importante per la validità, ma già catturato dall'eseguibilità per i casi estremi.

5. **Terminazioni/EOS (×0.5):** dettagli minori, coperti parzialmente dalle reward precedenti.

6. **No markdown (penalità):** non misura qualità VGDL, solo format compliance. È pura deterrenza.

---

## 5. Hyperparametri

### Parametri GRPO e motivazioni

| Parametro | Valore | Motivazione |
|---|---|---|
| `num_generations` | 8 | Più sample per prompt → stima group-relative più accurata (era 4) |
| `max_completion_length` | 800 | VGDL complessi possono essere lunghi; troncare causa parsing falliti artificiali (era 512) |
| `temperature` | 0.8 | Meno rumore rispetto a 0.9, mantenendo sufficiente esplorazione |
| `beta` | 0.04 | KL più forte previene reward hacking; 0.01 era troppo basso (era 0.01) |
| `num_train_epochs` | 5 | Dataset piccolo (180 esempi), più epoche necessarie (era 3) |
| `gradient_accumulation_steps` | 4 | Update più frequenti accelerano la convergenza (era 8) |
| `learning_rate` | 3e-6 | Più basso di prima per stabilità con reward_similarity (era 5e-6) |

### Il parametro beta (KL penalty)

Il termine KL nella loss GRPO misura quanto la policy corrente si è allontanata dal modello SFT di riferimento:

```
KL(π_θ || π_ref) = E[ log(π_θ(o|q) / π_ref(o|q)) ]
```

- **beta troppo basso (0.01):** il modello può degenerare verso output che massimizzano meccanicamente le reward ma perdono la coerenza del linguaggio. Esempio: output che contengono solo `BasicGame SpriteSet InteractionSet TerminationSet` guadagnano reward_structure ma sono vuoti.
- **beta troppo alto (>0.1):** il modello non impara nulla di nuovo rispetto al SFT, l'aggiornamento è troppo conservativo.
- **0.04:** bilanciamento ragionevole che permette apprendimento senza degenerazione.

### Il parametro num_generations (G)

Con G=4, la normalizzazione di gruppo ha poca varianza per casi in cui tutte le completions sono simili (es. tutte invalide early training). L'advantage diventa quasi zero → gradiente quasi nullo → apprendimento lento.

Con G=8, è più probabile che almeno alcune completions siano migliori delle altre anche early training, dando segnale di gradiente utile. Il costo computazionale aumenta ma è coperto da vLLM.

---

## 6. Flusso di training completo

### Dataset

Il dataset contiene 201 coppie `(descrizione, VGDL)`:
- **Train:** ~180 esempi (90%)
- **Test:** ~21 esempi (10%)

Ogni esempio viene formattato come:

```
<|im_start|>system
You are an expert in VGDL... Output ONLY raw VGDL code...
<|im_end|>
<|im_start|>user
{game_description}
<|im_end|>
<|im_start|>assistant
<think>

</think>
```

Il tag `<think>` è il prefix della completion. Il modello genera il codice VGDL subito dopo.

La colonna `vgdl` del dataset viene mantenuta (non rimossa) così da essere disponibile come `kwargs["vgdl"]` nelle reward functions.

### Caricamento del modello

```python
# 1. Carica Qwen3.5-4B con il LoRA SFT già applicato
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/qwen3.5/supervised-learning/16-32-0.05",
    ...
)

# 2. Abilita training dei parametri LoRA
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad_(True)
```

Il modello caricato è lo stesso SFT model che funge anche da `π_ref` nella KL penalty. Unsloth gestisce internamente il riferimento frozen.

### Iterazione di training

```
Per ogni epoca (5 epoche totali):
  Per ogni batch di prompt (effective batch size = 4):
    
    1. GENERAZIONE (vLLM):
       Genera 8 completions per ogni prompt nel batch
       (temperature=0.8, max_completion_length=800)
    
    2. REWARD EVALUATION:
       Per ogni completion:
         - reward_executability_shaped → chiama py-vgdl parser
         - reward_similarity          → Jaccard vs riferimento
         - reward_structure           → regex
         - reward_valid_sprite_classes → lookup set
         - reward_valid_interactions  → regex + lookup set
         - reward_valid_terminations  → regex + lookup set
         - reward_eos_boundary        → regex
         - reward_no_markdown         → regex
         R_i = somma di tutte le reward
    
    3. NORMALIZZAZIONE:
       A_i = (R_i - mean) / std  per i=1..8
    
    4. LOSS COMPUTATION:
       L = -mean(A_i * log π(c_i | prompt)) + 0.04 * KL
    
    5. GRADIENT UPDATE:
       backward() → optimizer step
       (aggiorna solo parametri LoRA)
    
  Salva checkpoint a fine epoca
```

### Struttura LoRA

Solo i parametri LoRA vengono aggiornati durante GRPO. Il modello base rimane frozen (4-bit quantizzato). Questo riduce drasticamente la memoria e il numero di parametri trainabili:

```
Base model parameters:  ~4B  (frozen, 4-bit)
LoRA parameters:        ~12M (trainabili, bf16)
Percentuale trainabile: ~0.3%
```

I moduli target includono tutte le proiezioni dell'attenzione e del FFN: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

---

## 7. Metriche

### Durante il training (log ogni 5 steps)

GRPOTrainer logga automaticamente:

| Metrica | Significato |
|---|---|
| `loss` | GRPO loss totale (negativo è buono) |
| `reward` | Reward media sul batch corrente |
| `reward_std` | Deviazione standard delle reward (alta = buona esplorazione) |
| `kl` | Divergenza KL dalla policy SFT di riferimento |
| `logps/chosen` | Log-probability delle completions selezionate |

### Cosa osservare

**Segnali di training sano:**
- `reward` cresce lentamente nel tempo
- `kl` rimane < 0.5 (se sale troppo, beta è troppo basso)
- `reward_std` non crolla a zero (il modello mantiene diversità)
- `loss` diminuisce o oscilla intorno a un valore stabile

**Segnali di problema:**
- `reward` piatta da subito → le reward functions non forniscono varianza nel gruppo (tutti 0 o tutti uguali)
- `kl` esplode → beta troppo basso, il modello si allontana troppo da SFT
- `loss` esplode → learning rate troppo alto
- `reward_std` → 0 → mode collapse (il modello genera sempre la stessa cosa)

### Valutazione finale

Dopo il training GRPO, il modello va valutato su test set con:

1. **Executability rate:** percentuale di output che passano il parser py-vgdl
2. **Jaccard similarity:** media della `vgdl_similarity` tra output e riferimento
3. Confronto con SFT baseline su entrambe le metriche

Script disponibili in `evaluation/`:
- `check_vgdl_executability.py`: validazione parser
- `eval_similarity.py`: Jaccard similarity
