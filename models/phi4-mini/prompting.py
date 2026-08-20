"""Shared prompt and output-cleaning helpers for the Phi-4-mini pipeline."""

from __future__ import annotations


def build_prompt(description: str) -> str:
    """Build the text-only instruction used by SFT and GRPO."""
    return f"""You are an expert VGDL (Video Game Description Language) programmer.
Generate a complete and syntactically valid VGDL game from the following natural-language description.

Rules:
- Output only the VGDL code, without explanations or Markdown fences.
- Include the complete BasicGame, SpriteSet, LevelMapping, InteractionSet and TerminationSet sections.
- Use valid VGDL syntax and indentation.

Game description:
{description.strip()}

VGDL:
BasicGame"""


def clean_generated_vgdl(text: str) -> str:
    """Extract VGDL code from a model response, tolerating Markdown wrappers."""
    fence = chr(96) * 3
    cleaned = text.strip()

    if cleaned.startswith(fence):
        first_newline = cleaned.find("\n")
        cleaned = cleaned[first_newline + 1 :] if first_newline >= 0 else ""

    cleaned = cleaned.rstrip()
    if cleaned.endswith(fence):
        cleaned = cleaned[: -len(fence)]

    start = cleaned.find("BasicGame")
    return cleaned[start:].strip() if start >= 0 else cleaned.strip()
