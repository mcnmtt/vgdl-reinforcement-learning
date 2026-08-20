"""
Reward functions for Phi-4-mini GRPO training on VGDL generation.

They mirror the Gemma4 reward design:
  - shaped VGDL executability
  - structural similarity against the reference VGDL
  - mandatory VGDL structure
  - valid sprite classes, interaction effects, termination conditions
  - correct EOS usage
  - markdown/prose penalty

Maximum theoretical reward: 9.5
Minimum theoretical reward: -0.5
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[2]
_EVAL_DIR = _REPO_ROOT / "evaluation"
_PYVGDL_DIR = _REPO_ROOT / "py-vgdl"

for _path in (_EVAL_DIR, _PYVGDL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from eval_similarity import vgdl_similarity as _vgdl_similarity

    _SIMILARITY_AVAILABLE = True
except ImportError:
    _SIMILARITY_AVAILABLE = False
    print("WARN: eval_similarity not found, reward_similarity disabled.")


# ========================
# VGDL ontology
# ========================

VALID_SPRITE_CLASSES = {
    "Immovable", "Passive", "ResourcePack", "Flicker", "Spreader",
    "Conveyor", "Missile", "OrientedFlicker", "Walker", "WalkJumper",
    "RandomNPC", "RandomInertial", "RandomMissile", "ErraticMissile",
    "Bomber", "Chaser", "Fleeing", "AStarChaser",
    "SpawnPoint", "Portal",
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
INVALID_BOUNDARY_KEYWORDS = {"edge", "screen", "wall_bound", "boundary"}

_RE_SECTION = {
    section: re.compile(rf"^\s*{section}\b", re.MULTILINE)
    for section in MANDATORY_SECTIONS
}
_RE_SPRITE_CLASS = re.compile(r">\s*([A-Z][A-Za-z]+)")
_RE_INTERACTION_SECTION = re.compile(
    r"InteractionSet(.*?)(?=TerminationSet|LevelMapping|SpriteSet|\Z)",
    re.DOTALL,
)
_RE_EFFECT = re.compile(r">\s*([a-z][A-Za-z]+)")
_RE_TERMINATION_SECTION = re.compile(
    r"TerminationSet(.*?)(?=InteractionSet|LevelMapping|SpriteSet|\Z)",
    re.DOTALL,
)
_RE_TERMINATION_CONDITION = re.compile(r"^\s+([A-Z][A-Za-z]+)", re.MULTILINE)
_RE_EOS = re.compile(r"\bEOS\b")
_RE_MARKDOWN = re.compile(r"```")
_RE_PROSE_PREFIX = re.compile(
    r"^(Here|This|The game|I have|Sure|Below|Let me|I'll)",
    re.IGNORECASE,
)

_VGDL_PARSER = None


def _completion_text(completion: Any) -> str:
    """Normalize TRL completion formats to plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, list):
                    parts.extend(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict)
                    )
                else:
                    parts.append(str(content))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _texts(completions) -> list[str]:
    return [_completion_text(completion).strip() for completion in completions]


def _get_parser():
    global _VGDL_PARSER
    if _VGDL_PARSER is None:
        from vgdl.core import VGDLParser  # type: ignore

        _VGDL_PARSER = VGDLParser()
    return _VGDL_PARSER


def _parse_vgdl_string(vgdl_str: str):
    try:
        game = _get_parser().parseGame(vgdl_str.strip())
        return game, []
    except Exception as exc:
        return None, [f"Parsing error: {exc}"]


def _validate_vgdl_string(vgdl_str: str):
    game, errors = _parse_vgdl_string(vgdl_str)
    if game is None:
        return False, errors

    try:
        sprites = set(game.sprite_constr.keys())
    except Exception:
        sprites = set()

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

    try:
        for term in game.terminations:
            if hasattr(term, "stype") and term.stype and term.stype not in sprites:
                errors.append(f"Termination references undefined sprite: {term.stype}")
    except Exception:
        pass

    return len(errors) == 0, errors


def reward_executability_shaped(completions, **kwargs):
    """Weighted shaped reward for parser/semantic validity."""
    weight = 3.0
    rewards = []
    for text in _texts(completions):
        valid, errors = _validate_vgdl_string(text)
        if valid:
            rewards.append(weight)
            continue

        score = 0.0
        if text.startswith("BasicGame"):
            score += 0.1
        for section in MANDATORY_SECTIONS:
            if _RE_SECTION[section].search(text):
                score += 0.1
        if errors and not any("Parsing error" in error for error in errors):
            score += 0.3
        rewards.append(min(score, 0.95) * weight)
    return rewards


def reward_similarity(completions, **kwargs):
    """Weighted structural Jaccard similarity against dataset reference VGDL."""
    weight = 2.0
    if not _SIMILARITY_AVAILABLE:
        return [0.0] * len(completions)

    references = kwargs.get("vgdl", [None] * len(completions))
    rewards = []
    for comp, ref in zip(_texts(completions), references):
        if not ref:
            rewards.append(0.0)
            continue
        try:
            result = _vgdl_similarity(comp, ref)
            rewards.append(result["final_score"] * weight)
        except Exception:
            rewards.append(0.0)
    return rewards


def reward_structure(completions, **kwargs):
    """Weighted reward for BasicGame and mandatory sections."""
    weight = 1.5
    rewards = []
    for text in _texts(completions):
        score = 0.0
        if text.startswith("BasicGame"):
            score += 0.2
        for section in MANDATORY_SECTIONS:
            if _RE_SECTION[section].search(text):
                score += 0.2
        rewards.append(score * weight)
    return rewards


def reward_valid_sprite_classes(completions, **kwargs):
    """Weighted fraction of valid SpriteSet class names."""
    weight = 1.0
    rewards = []
    for text in _texts(completions):
        matches = _RE_SPRITE_CLASS.findall(text)
        if not matches:
            rewards.append(0.5 * weight)
            continue
        valid_count = sum(cls in VALID_SPRITE_CLASSES for cls in matches)
        rewards.append((valid_count / len(matches)) * weight)
    return rewards


def reward_valid_interactions(completions, **kwargs):
    """Weighted fraction of valid InteractionSet effects."""
    weight = 1.0
    rewards = []
    for text in _texts(completions):
        inter_match = _RE_INTERACTION_SECTION.search(text)
        if not inter_match:
            rewards.append(0.5 * weight)
            continue
        effects = _RE_EFFECT.findall(inter_match.group(1))
        if not effects:
            rewards.append(0.5 * weight)
            continue
        valid_count = sum(effect in VALID_INTERACTION_EFFECTS for effect in effects)
        rewards.append((valid_count / len(effects)) * weight)
    return rewards


def reward_valid_terminations(completions, **kwargs):
    """Weighted fraction of valid TerminationSet condition classes."""
    weight = 0.5
    rewards = []
    for text in _texts(completions):
        term_match = _RE_TERMINATION_SECTION.search(text)
        if not term_match:
            rewards.append(0.5 * weight)
            continue
        conditions = _RE_TERMINATION_CONDITION.findall(term_match.group(1))
        if not conditions:
            rewards.append(0.5 * weight)
            continue
        valid_count = sum(cond in VALID_TERMINATION_CONDITIONS for cond in conditions)
        rewards.append((valid_count / len(conditions)) * weight)
    return rewards


def reward_eos_boundary(completions, **kwargs):
    """Weighted reward for using EOS instead of invalid boundary words."""
    weight = 0.5
    rewards = []
    for text in _texts(completions):
        has_eos = bool(_RE_EOS.search(text))
        has_invalid = any(
            bool(re.search(rf"\b{kw}\b", text, re.IGNORECASE))
            for kw in INVALID_BOUNDARY_KEYWORDS
        )
        if has_invalid:
            rewards.append(0.0)
        elif has_eos:
            rewards.append(weight)
        else:
            rewards.append(0.5 * weight)
    return rewards


def reward_no_markdown(completions, **kwargs):
    """Negative-only penalty for markdown fences or prose preambles."""
    rewards = []
    for text in _texts(completions):
        if _RE_MARKDOWN.search(text):
            rewards.append(-0.5)
        elif _RE_PROSE_PREFIX.match(text):
            rewards.append(-0.2)
        else:
            rewards.append(0.0)
    return rewards


REWARD_FUNCTIONS = [
    reward_executability_shaped,
    reward_similarity,
    reward_structure,
    reward_valid_sprite_classes,
    reward_valid_interactions,
    reward_valid_terminations,
    reward_eos_boundary,
    reward_no_markdown,
]
