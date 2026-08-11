"""
Reward functions per il training GRPO su generazione VGDL.

Valutano le completions generate dal modello su diversi criteri,
con pesi proporzionali alla loro importanza:
  - Eseguibilita' VGDL shaped  (peso x3.0) — critica
  - Similarita' Jaccard        (peso x2.0) — semantica
  - Struttura corretta         (peso x1.5) — strutturale
  - Classi sprite valide       (peso x1.0) — ontologia
  - Effetti interazione validi (peso x1.0) — ontologia
  - Condizioni terminazione    (peso x0.5) — ontologia
  - Uso corretto EOS           (peso x0.5) — convenzione
  - No markdown/prosa          (penalita') — formato

Totale massimo teorico: 9.5
Totale minimo teorico: -0.5
"""

import os
import re
import sys

# Setup path per importare eval_similarity dalla directory evaluation/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", "evaluation"))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

try:
    from eval_similarity import vgdl_similarity as _vgdl_similarity
    _SIMILARITY_AVAILABLE = True
except ImportError:
    _SIMILARITY_AVAILABLE = False
    print("WARN: eval_similarity non trovato, reward_similarity disabilitata.")


# ========================
# Ontologia VGDL (estratta da py-vgdl/vgdl/ontology.py)
# ========================

VALID_SPRITE_CLASSES = {
    # Sprite statici / immovable
    "Immovable", "Passive", "ResourcePack", "Flicker", "Spreader",
    # Sprite orientati
    "Conveyor", "Missile", "OrientedFlicker", "Walker", "WalkJumper",
    # NPC
    "RandomNPC", "RandomInertial", "RandomMissile", "ErraticMissile",
    "Bomber", "Chaser", "Fleeing", "AStarChaser",
    # Producer
    "SpawnPoint", "Portal",
    # Avatar (player-controlled)
    "MovingAvatar", "HorizontalAvatar", "VerticalAvatar", "FlakAvatar",
    "OrientedAvatar", "RotatingAvatar", "RotatingFlippingAvatar",
    "NoisyRotatingFlippingAvatar", "ShootAvatar", "AimedAvatar",
    "AimedFlakAvatar", "InertialAvatar", "MarioAvatar",
}

VALID_INTERACTION_EFFECTS = {
    "killSprite", "cloneSprite", "transformTo",
    "stepBack", "undoAll",
    "bounceForward", "bounceDirection", "wallBounce", "wallStop",
    "conveySprite", "windGust", "slipForward", "pullWithIt",
    "attractGaze", "turnAround", "reverseDirection", "flipDirection",
    "wrapAround", "teleportToExit",
    "killIfSlow", "killIfFromAbove", "killIfAlive",
    "killIfHasMore", "killIfHasLess",
    "killIfOtherHasMore", "killIfOtherHasLess",
    "collectResource", "changeResource", "spawnIfHasMore",
}

VALID_TERMINATION_CONDITIONS = {
    "Timeout", "SpriteCounter", "MultiSpriteCounter",
}

MANDATORY_SECTIONS = ["SpriteSet", "LevelMapping", "InteractionSet", "TerminationSet"]

# Keyword NON valide per il bordo dello schermo (si deve usare EOS)
INVALID_BOUNDARY_KEYWORDS = {"edge", "screen", "wall_bound", "boundary"}


# ========================
# Validazione VGDL su stringa
# (adattamento di evaluation/check_vgdl_executability.py)
# ========================

def _parse_vgdl_string(vgdl_str: str):
    """Parsa una stringa VGDL. Restituisce (game | None, errori: list)."""
    try:
        from vgdl.core import VGDLParser  # type: ignore
        parser = VGDLParser()
        game = parser.parseGame(vgdl_str.strip())
        return game, []
    except Exception as e:
        return None, [f"Parsing error: {e}"]


def _validate_vgdl_string(vgdl_str: str):
    """Valida una stringa VGDL. Restituisce (valid: bool, errori: list)."""
    game, errors = _parse_vgdl_string(vgdl_str)
    if game is None:
        return False, errors

    try:
        sprites = set(game.sprite_constr.keys())
    except Exception:
        sprites = set()

    # Controlla sprite nelle interazioni
    try:
        valid_sprites = set(game.sprite_constr.keys())
        special_sprites = {"EOS", "wall"}
        for _, (_, _, stypes) in game.sprite_constr.items():
            valid_sprites.update(stypes)
        for interaction in game.collision_eff:
            s1, s2 = interaction[0], interaction[1]
            if s1 not in valid_sprites and s1 not in special_sprites:
                errors.append(f"Interaction uses undefined sprite: {s1}")
            if s2 not in valid_sprites and s2 not in special_sprites:
                errors.append(f"Interaction uses undefined sprite: {s2}")
    except Exception:
        pass

    # Controlla sprite nelle terminazioni
    try:
        for term in game.terminations:
            if hasattr(term, "stype") and term.stype and term.stype not in sprites:
                errors.append(f"Termination references undefined sprite: {term.stype}")
    except Exception:
        pass

    return len(errors) == 0, errors


# ========================
# Reward Functions
# ========================

def reward_executability_shaped(completions, **kwargs):
    """
    Reward principale (peso x3.0): credito parziale basato su quanto il VGDL
    e' vicino alla validita', invece di un binario 0/1.

    Scala:
      3.0        — VGDL completamente valido (parser + check semantici OK)
      0.3 – 2.85 — credito parziale per struttura parzialmente corretta
      0.0        — nessuna struttura riconoscibile
    """
    WEIGHT = 3.0
    rewards = []
    for text in completions:
        valid, errors = _validate_vgdl_string(text)
        if valid:
            rewards.append(1.0 * WEIGHT)
            continue
        # Credito parziale: struttura presente
        score = 0.0
        if text.strip().startswith("BasicGame"):
            score += 0.1
        for section in MANDATORY_SECTIONS:
            if re.search(rf'^\s*{section}\b', text, re.MULTILINE):
                score += 0.1  # +0.4 max per le 4 sezioni obbligatorie
        # Bonus: parsing riuscito ma errori semantici (sprite non definiti, ecc.)
        # Significa che la struttura sintattica e' corretta → credito extra
        if errors and not any("Parsing error" in e for e in errors):
            score += 0.3
        # Cap a 0.95 cosi' il reward e' sempre < 3.0 se non completamente valido
        rewards.append(min(score, 0.95) * WEIGHT)
    return rewards


def reward_similarity(completions, **kwargs):
    """
    Reward semantica (peso x2.0): Jaccard similarity tra il VGDL generato
    e il VGDL di riferimento del dataset (colonna 'vgdl').

    Usa tre componenti pesate (da eval_similarity.py):
      - Sprite similarity      (25%)
      - Interaction similarity (50%) — componente piu' importante
      - Termination similarity (25%)

    Richiede che la colonna 'vgdl' sia presente nel dataset (non rimossa).
    Restituisce 0.0 se il riferimento non e' disponibile.
    """
    WEIGHT = 2.0
    if not _SIMILARITY_AVAILABLE:
        return [0.0] * len(completions)
    references = kwargs.get("vgdl", [None] * len(completions))
    rewards = []
    for comp, ref in zip(completions, references):
        if not ref:
            rewards.append(0.0)
            continue
        try:
            result = _vgdl_similarity(comp, ref)
            rewards.append(result["final_score"] * WEIGHT)
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_structure(completions, **kwargs):
    """
    Reward struttura (peso x1.5): il VGDL ha le sezioni obbligatorie?
      +0.2  se inizia con 'BasicGame'
      +0.2  per ogni sezione obbligatoria presente (x4 = +0.8)
    Massimo: 1.0 x 1.5 = 1.5
    """
    WEIGHT = 1.5
    rewards = []
    for text in completions:
        score = 0.0
        if text.strip().startswith("BasicGame"):
            score += 0.2
        for section in MANDATORY_SECTIONS:
            if re.search(rf'^\s*{section}\b', text, re.MULTILINE):
                score += 0.2
        rewards.append(score * WEIGHT)
    return rewards


def reward_valid_sprite_classes(completions, **kwargs):
    """
    Reward ontologia sprite (peso x1.0):
    Frazione di '> ClassName' che corrispondono a classi sprite valide.
    Score neutro 0.5 se nessuna classe trovata.
    """
    WEIGHT = 1.0
    rewards = []
    for text in completions:
        matches = re.findall(r'>\s*([A-Z][A-Za-z]+)', text)
        if not matches:
            rewards.append(0.5 * WEIGHT)
            continue
        valid_count = sum(1 for cls in matches if cls in VALID_SPRITE_CLASSES)
        rewards.append((valid_count / len(matches)) * WEIGHT)
    return rewards


def reward_valid_interactions(completions, **kwargs):
    """
    Reward ontologia interazioni (peso x1.0):
    Frazione di effetti nell'InteractionSet che sono effetti validi.
    Score neutro 0.5 se sezione non trovata.
    """
    WEIGHT = 1.0
    rewards = []
    for text in completions:
        inter_match = re.search(
            r'InteractionSet(.*?)(?=TerminationSet|LevelMapping|SpriteSet|\Z)',
            text, re.DOTALL
        )
        if not inter_match:
            rewards.append(0.5 * WEIGHT)
            continue
        section_text = inter_match.group(1)
        # Effetti iniziano con lettera minuscola dopo ">"
        effects = re.findall(r'>\s*([a-z][A-Za-z]+)', section_text)
        if not effects:
            rewards.append(0.5 * WEIGHT)
            continue
        valid_count = sum(1 for e in effects if e in VALID_INTERACTION_EFFECTS)
        rewards.append((valid_count / len(effects)) * WEIGHT)
    return rewards


def reward_valid_terminations(completions, **kwargs):
    """
    Reward ontologia terminazioni (peso x0.5):
    Frazione di condizioni di terminazione valide (Timeout, SpriteCounter,
    MultiSpriteCounter).
    Score neutro 0.5 se nessuna condizione trovata.
    """
    WEIGHT = 0.5
    rewards = []
    for text in completions:
        term_match = re.search(
            r'TerminationSet(.*?)(?=InteractionSet|LevelMapping|SpriteSet|\Z)',
            text, re.DOTALL
        )
        if not term_match:
            rewards.append(0.5 * WEIGHT)
            continue
        section_text = term_match.group(1)
        # Condizioni iniziano con lettera maiuscola (indentate)
        conditions = re.findall(r'^\s+([A-Z][A-Za-z]+)', section_text, re.MULTILINE)
        if not conditions:
            rewards.append(0.5 * WEIGHT)
            continue
        valid_count = sum(1 for c in conditions if c in VALID_TERMINATION_CONDITIONS)
        rewards.append((valid_count / len(conditions)) * WEIGHT)
    return rewards


def reward_eos_boundary(completions, **kwargs):
    """
    Reward convenzione EOS (peso x0.5):
      0.5  (1.0 x peso) — usa EOS correttamente
      0.0              — usa keyword non valide (edge, screen, wall_bound, boundary)
      0.25 (0.5 x peso)— non menziona bordi (neutro)
    """
    WEIGHT = 0.5
    rewards = []
    for text in completions:
        has_eos = bool(re.search(r'\bEOS\b', text))
        has_invalid = any(
            bool(re.search(rf'\b{kw}\b', text, re.IGNORECASE))
            for kw in INVALID_BOUNDARY_KEYWORDS
        )
        if has_invalid:
            rewards.append(0.0)
        elif has_eos:
            rewards.append(1.0 * WEIGHT)
        else:
            rewards.append(0.5 * WEIGHT)
    return rewards


def reward_no_markdown(completions, **kwargs):
    """
    Penalita' formato (solo negativa o zero):
    Penalizza output che contengono markdown o prosa invece di VGDL puro.
      -0.5  se contiene backtick/fences markdown (```)
      -0.2  se inizia con prosa introduttiva
       0.0  altrimenti (nessuna penalita')
    """
    rewards = []
    for text in completions:
        stripped = text.strip()
        if re.search(r'```', stripped):
            rewards.append(-0.5)
        elif re.match(
            r'^(Here|This|The game|I have|Sure|Below|Let me|I\'ll)',
            stripped, re.IGNORECASE
        ):
            rewards.append(-0.2)
        else:
            rewards.append(0.0)
    return rewards


# Lista ordinata di reward functions (GRPOTrainer le somma automaticamente)
#
# Riepilogo pesi e range:
#   reward_executability_shaped : range [0.0,  3.0]  — CRITICA
#   reward_similarity           : range [0.0,  2.0]  — semantica
#   reward_structure            : range [0.0,  1.5]  — struttura
#   reward_valid_sprite_classes : range [0.5,  1.0]  — ontologia sprite
#   reward_valid_interactions   : range [0.5,  1.0]  — ontologia interazioni
#   reward_valid_terminations   : range [0.25, 0.5]  — ontologia terminazioni
#   reward_eos_boundary         : range [0.0,  0.5]  — convenzione EOS
#   reward_no_markdown          : range [-0.5, 0.0]  — formato (solo penalita')
#
# Totale massimo teorico: 9.5
# Totale minimo teorico: -0.5

REWARD_FUNCTIONS = [
    reward_executability_shaped,    # peso x3.0 — CRITICA
    reward_similarity,              # peso x2.0 — semantica
    reward_structure,               # peso x1.5 — struttura
    reward_valid_sprite_classes,    # peso x1.0 — ontologia
    reward_valid_interactions,      # peso x1.0 — ontologia
    reward_valid_terminations,      # peso x0.5 — ontologia
    reward_eos_boundary,            # peso x0.5 — convenzione
    reward_no_markdown,             # penalita' — formato
]
