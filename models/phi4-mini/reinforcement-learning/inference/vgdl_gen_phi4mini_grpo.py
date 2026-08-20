"""
Inference script for the GRPO-fine-tuned Phi-4-mini-instruct model.

The GRPO training produced a LoRA adapter saved to:
    models/phi4-mini/reinforcement-learning/grpo-output-sft/last-model

This script loads the base model (microsoft/Phi-4-mini-instruct) and merges
the adapter on top, then generates VGDL from a natural language description.

Usage examples
--------------
# Single description file
python models/phi4-mini/reinforcement-learning/inference/vgdl_gen_phi4mini_grpo.py \
    --description-file dataset/descriptions/1_sokoban.txt

# Batch over first 5 descriptions
python models/phi4-mini/reinforcement-learning/inference/vgdl_gen_phi4mini_grpo.py \
    --limit 5

# All descriptions
python models/phi4-mini/reinforcement-learning/inference/vgdl_gen_phi4mini_grpo.py \
    --limit 0

# Compare specific checkpoint instead of last-model
python models/phi4-mini/reinforcement-learning/inference/vgdl_gen_phi4mini_grpo.py \
    --adapter-path models/phi4-mini/reinforcement-learning/grpo-output-sft/checkpoint-135 \
    --description-file dataset/descriptions/1_sokoban.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ======================
# Paths
# ======================
BASE_MODEL_NAME = "microsoft/Phi-4-mini-instruct"
_THIS_DIR = Path(__file__).resolve().parent
_PHI_DIR = _THIS_DIR.parents[1]
_REPO_ROOT = _PHI_DIR.parents[1]
sys.path.insert(0, str(_PHI_DIR))

from prompting import build_prompt  # noqa: E402

DEFAULT_ADAPTER_PATH = (
    _PHI_DIR / "reinforcement-learning" / "grpo-output-sft" / "last-model"
)

# ======================
# Experiment Config
# ======================
DESCRIPTION_DIR = _REPO_ROOT / "dataset" / "descriptions"
OUTPUT_DIR = _PHI_DIR / "reinforcement-learning" / "vgdl-sft-start"
MAX_DESCRIPTIONS: int | None = 3
MAX_NEW_TOKENS = 400
VGDL_PREFIX = "BasicGame"

FEW_SHOT_EXAMPLE = """
Example 1 - Basic sprites and interactions:
Description:
Player role: The player controls a MovingAvatar.
Entities:
- hole: an Immovable static object with color=DARKBLUE.
- avatar: a MovingAvatar.
- box: a Passive object.
- wall: an Immovable static object.
Interactions:
- avatar wall > stepBack: avatar cannot pass through walls.
- box avatar > bounceForward: boxes are pushed in the direction the avatar is moving.
- box wall > undoAll: boxes cannot pass through walls.
- box hole > killSprite: boxes that enter holes are destroyed.
Win condition: SpriteCounter stype=box limit=0 win=True.
Lose condition: None.
Objective: Push all boxes into holes.

VGDL:
BasicGame
    SpriteSet
        hole > Immovable color=DARKBLUE
        avatar > MovingAvatar
        box > Passive
        wall > Immovable
    LevelMapping
        0 > hole
        1 > box
    InteractionSet
        avatar wall > stepBack
        box avatar > bounceForward
        box wall > undoAll
        box hole > killSprite
    TerminationSet
        SpriteCounter stype=box limit=0 win=True

Example 2 - Child sprites (shooter + projectile):
Description:
Player role: The player controls a ShootAvatar that shoots bullet projectiles forward.
Entities:
- avatar: a ShootAvatar. bullet is its child sprite and is spawned when shooting.
- bullet: a Missile, child of avatar. Parameters are written inline: bullet > Missile speed=2.
- enemy: a Passive object.
- wall: an Immovable static object.
Interactions:
- bullet wall > killSprite: bullets are destroyed when hitting walls.
- bullet enemy > killSprite: enemies are destroyed when hit by bullets.
- avatar wall > stepBack: avatar cannot pass through walls.
- bullet EOS > killSprite: bullets are destroyed when reaching the edge of the screen (EOS).
Win condition: SpriteCounter stype=enemy limit=0 win=True.
Lose condition: SpriteCounter stype=avatar limit=0 win=False.
Objective: Shoot all enemies to win.

VGDL:
BasicGame
    SpriteSet
        avatar > ShootAvatar
            bullet > Missile speed=2
        enemy > Passive
        wall > Immovable
    LevelMapping
        0 > enemy
        1 > wall
    InteractionSet
        avatar wall > stepBack
        bullet wall > killSprite
        bullet enemy > killSprite
        bullet EOS > killSprite
    TerminationSet
        SpriteCounter stype=enemy limit=0 win=True
        SpriteCounter stype=avatar limit=0 win=False

Example 3 - Sprite groups and EOS:
Description:
Player role: The player controls a MarioAvatar with physicstype=GravityPhysics.
Entities:
- projectile: a Missile with orientation=RIGHT and speed=2.
- moving: a sprite group with physicstype=GravityPhysics, containing avatar and npc.
- avatar: a MarioAvatar, member of the moving group.
- npc: a Walker, member of the moving group, with orientation=LEFT.
- goal: an Immovable static object.
- wall: an Immovable static object.
Interactions:
- avatar npc > killIfFromAbove: if avatar arrives from above, npc is destroyed.
- npc avatar > killIfAlive: if npc is alive and collides with avatar, avatar is destroyed.
- moving EOS > killSprite: any sprite in the moving group that reaches EOS is destroyed.
- moving wall > wallStop friction=0.1: moving sprites stop at walls with friction=0.1.
- goal avatar > killSprite: when avatar reaches goal, goal is destroyed.
Win condition: SpriteCounter stype=goal limit=0 win=True.
Lose condition: SpriteCounter stype=avatar limit=0 win=False.
Objective: Reach the goal while avoiding enemies.

VGDL:
BasicGame
    SpriteSet
        projectile > Missile orientation=RIGHT speed=2
        moving > physicstype=GravityPhysics
            avatar > MarioAvatar
            npc > Walker orientation=LEFT
        goal > Immovable
        wall > Immovable
    LevelMapping
        G > goal
        1 > npc
    InteractionSet
        avatar npc > killIfFromAbove
        npc avatar > killIfAlive
        moving EOS > killSprite
        moving wall > wallStop friction=0.1
        goal avatar > killSprite
    TerminationSet
        SpriteCounter stype=goal limit=0 win=True
        SpriteCounter stype=avatar limit=0 win=False

Example 4 - Nested sprite groups:
Description:
Player role: The player controls a MarioAvatar, member of the moving group.
Entities:
- moving: a sprite group with physicstype=GravityPhysics, containing avatar and enemies.
- avatar: a MarioAvatar with airsteering=True, member of the moving group.
- enemies: a sprite subgroup of moving with orientation=LEFT, containing walker and jumper.
- walker: a Walker, member of the enemies subgroup.
- jumper: a WalkJumper, member of the enemies subgroup.
- wall: an Immovable static object.
Interactions:
- enemies avatar > killIfFromAbove: if avatar arrives from above, the enemy is destroyed.
- moving EOS > killSprite: any moving sprite that reaches EOS is destroyed.
- moving wall > wallStop friction=0.1: moving sprites stop at walls with friction=0.1.
Win condition: SpriteCounter stype=avatar limit=0 win=False.
Lose condition: None.
Objective: Avoid enemies and walls.

VGDL:
BasicGame
    SpriteSet
        moving > physicstype=GravityPhysics
            avatar > MarioAvatar airsteering=True
            enemies > orientation=LEFT
                walker > Walker
                jumper > WalkJumper
        wall > Immovable
    LevelMapping
        1 > walker
        2 > jumper
    InteractionSet
        enemies avatar > killIfFromAbove
        moving EOS > killSprite
        moving wall > wallStop friction=0.1
    TerminationSet
        SpriteCounter stype=avatar limit=0 win=False
""".strip()


def build_legacy_few_shot_prompt(game_description: str) -> str:
    return f"""You are an expert in VGDL (Video Game Description Language).
Convert the game description below into a valid VGDL specification.

STRICT RULES:
- Only include mechanics explicitly described. Do NOT invent entities or rules.
- Always include all four sections in this exact order: SpriteSet, LevelMapping, InteractionSet, TerminationSet.
- Output ONLY raw VGDL code. No explanation, no markdown, no comments.
- Each sprite must have EXACTLY ONE class. Never concatenate multiple class names.
- Sub-indentation inside SpriteSet is ONLY used for child sprites (e.g. bullet indented under ShootAvatar). Never use it for properties.
- Sprite groups (e.g. moving, evil) that have no class of their own are defined by indenting their children under them WITHOUT a class name. Example:
      moving >
          avatar > MarioAvatar
- YOU MUST USE ONLY the sprite classes, interaction effects, and termination conditions listed below, do NOT INVENT NEW ONES.

VALID SPRITE CLASSES (use ONLY these, one per sprite):
- MovingAvatar             -> 4-directional player avatar
- HorizontalAvatar         -> player avatar that moves left/right only
- VerticalAvatar           -> player avatar that moves up/down only
- OrientedAvatar           -> player avatar that retains orientation
- FlakAvatar               -> horizontal avatar that shoots upward (child sprite required)
- ShootAvatar              -> oriented avatar that shoots forward (child sprite required)
- AimedAvatar              -> player avatar that can change the direction of firing, but not move
- AimedFlakAvatar          -> horizontal avatar with aim control and left/right movement (child sprite required)
- InertialAvatar           -> avatar with continuous inertial physics
- MarioAvatar              -> avatar with gravity and jump mechanics
- RotatingAvatar           -> avatar that rotates and moves forward/backward relative to its orientation
- RotatingFlippingAvatar   -> like RotatingAvatar but DOWN rotates 180 degrees
- NoisyRotatingFlippingAvatar -> RotatingFlippingAvatar with stochastic noise (noiseLevel=0.1)
- Immovable                -> static object, cannot be moved (walls, goals, holes)
- Passive                  -> object that can be pushed or interacted with
- ResourcePack             -> collectible resource object
- Flicker                  -> sprite that disappears after a few timesteps
- OrientedFlicker          -> short-lived directional sprite (e.g. sword slash)
- Missile                  -> moves continuously in a fixed direction
- RandomMissile            -> missile with randomized direction and speed at initialization
- ErraticMissile           -> missile that randomly changes direction with probability prob=
- Bomber                   -> missile that also spawns sprites periodically (requires stype=)
- Walker                   -> moves horizontally, bounces off walls
- WalkJumper               -> moves horizontally and occasionally jumps
- RandomNPC                -> NPC that moves randomly each step
- RandomInertial           -> oriented sprite with continuous physics that moves randomly
- Chaser                   -> NPC that moves toward a target sprite type (requires stype=)
- Fleeing                  -> NPC that moves away from a target sprite type (requires stype=)
- AStarChaser              -> NPC that uses A* search to chase a target sprite type (requires stype=)
- Spreader                 -> spreads to adjacent cells over time
- Conveyor                 -> static object that moves other sprites along its orientation
- Portal                   -> teleports sprites that touch it (requires stype=)
- SpawnPoint               -> spawns sprites of a given type over time (requires stype=)

VALID INTERACTION EFFECTS (use ONLY these):
- killSprite               -> destroy the first sprite
- cloneSprite              -> clone the first sprite
- transformTo              -> replace first sprite with another type (MUST include stype=ClassName)
- stepBack                 -> undo the move of the first sprite
- undoAll                  -> undo last move of all sprites
- bounceForward            -> push first sprite in the partner's last direction
- bounceDirection          -> bounce first sprite based on center-to-center direction
- conveySprite             -> move first sprite along partner's orientation
- windGust                 -> like conveySprite but with stochastic force variation
- slipForward              -> stochastically move first sprite forward along its orientation
- attractGaze              -> stochastically rotate first sprite's orientation toward partner's
- turnAround               -> make first sprite reverse direction (stepBack + reverseDirection)
- reverseDirection         -> reverse the orientation of the first sprite
- flipDirection            -> set orientation of first sprite randomly
- wallBounce               -> bounce first sprite off wall orthogonally
- wallStop                 -> stop first sprite at wall, slide along it
- killIfSlow               -> kill first sprite if relative speed is below threshold
- killIfFromAbove          -> kill first sprite only if partner arrived from above
- killIfAlive              -> kill first sprite only if partner is not already in kill list
- killIfHasMore            -> kill first sprite if it has more than limit of a resource (requires resource= limit=)
- killIfHasLess            -> kill first sprite if it has less than limit of a resource (requires resource= limit=)
- killIfOtherHasMore       -> kill first sprite if partner has more than limit of a resource (requires resource= limit=)
- killIfOtherHasLess       -> kill first sprite if partner has less than limit of a resource (requires resource= limit=)
- wrapAround               -> wrap first sprite to opposite side of screen
- pullWithIt               -> carry first sprite along with partner's movement
- teleportToExit           -> teleport first sprite to a portal/exit
- collectResource          -> collect a resource pack into partner's inventory
- changeResource           -> increment a specific resource in first sprite (requires resource= value=)
- spawnIfHasMore           -> spawn a sprite if first sprite has more than limit of a resource (requires resource= stype= limit=)

VALID TERMINATION CONDITIONS (use ONLY these):
- SpriteCounter stype=<n> limit=<n> win=True/False
- MultiSpriteCounter stype1=<n> stype2=<n> limit=<n> win=True/False
- Timeout limit=<n> win=True/False   (limit must be > 0)

EDGE OF SCREEN - CRITICAL RULE:
- The edge of the screen is represented as the keyword EOS (NOT "edge", "screen", "border", or any other word).
- Correct: "avatar EOS > killSprite"
- WRONG: "avatar edge > killSprite", "avatar screen > killSprite"

FORBIDDEN:
- Any sprite class, interaction effect, or termination type not listed above.
- Concatenated class names (e.g. WalkerMissile is INVALID).
- Sub-indented property lines in SpriteSet (only child sprites may be indented).
- transformTo without stype=.
- Timeout with limit=0.
- Using "edge", "screen", or "border" instead of EOS.

{FEW_SHOT_EXAMPLE}

Now convert this description:
Description:
{game_description.strip()}

Respond with ONLY the raw VGDL code block, starting with 'BasicGame'. No explanation, no markdown fences, no comments, no reasoning.

VGDL:
BasicGame"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VGDL generation with the GRPO-fine-tuned Phi-4-mini-instruct LoRA adapter."
    )
    parser.add_argument(
        "--adapter-path",
        default=str(DEFAULT_ADAPTER_PATH),
        help=(
            "Path to the PEFT/LoRA adapter directory. "
            f"Defaults to '{DEFAULT_ADAPTER_PATH}'."
        ),
    )
    parser.add_argument(
        "--description-file",
        default=None,
        help=(
            "Single text file containing a natural language game description. "
            "If omitted, the script runs in batch mode over --description-dir."
        ),
    )
    parser.add_argument(
        "--description-text",
        default=None,
        help=(
            "Inline game description string (alternative to --description-file). "
            "Takes priority over --description-file."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to save the generated VGDL in single-file mode.",
    )
    parser.add_argument(
        "--description-dir",
        default=str(DESCRIPTION_DIR),
        help="Directory containing dataset descriptions for batch mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory where batch-mode VGDL outputs are saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Override MAX_DESCRIPTIONS for batch mode. Use 0 for all. "
            "If omitted, the in-code MAX_DESCRIPTIONS value is used."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected inputs and the first prompt without loading the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load without 4-bit quantization. Needs much more VRAM/RAM.",
    )
    parser.add_argument(
        "--gpu-max-memory",
        default="4GiB",
        help="Maximum GPU memory passed to accelerate device_map.",
    )
    parser.add_argument(
        "--cpu-max-memory",
        default="24GiB",
        help="Maximum CPU memory passed to accelerate device_map.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation. Use eager on consumer GPUs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Use the model repository's custom code.",
    )
    parser.add_argument(
        "--prompt-mode",
        default="raw",
        choices=["raw", "chat"],
        help="Use raw prompt text or the model chat template.",
    )
    parser.add_argument(
        "--merge-adapter",
        action="store_true",
        help=(
            "Merge LoRA weights into the base model before inference. "
            "Slightly faster generation but uses more memory."
        ),
    )
    parser.add_argument(
        "--disable-adapter",
        action="store_true",
        help=(
            "Diagnostic option: load the adapter files but temporarily disable "
            "the LoRA adapter during generation. This should match zero-shot."
        ),
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help=(
            "Diagnostic option: for each input, also print the output obtained "
            "with the LoRA adapter disabled before printing the GRPO output."
        ),
    )
    return parser.parse_args()


def read_description(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def numeric_sort_key(path: Path) -> tuple[int, str]:
    prefix = path.stem.split("_", 1)[0]
    try:
        return int(prefix), path.name
    except ValueError:
        return 10**9, path.name


def select_description_files(args: argparse.Namespace) -> list[Path]:
    if args.description_text or args.description_file:
        return []  # handled separately

    description_dir = Path(args.description_dir)
    files = sorted(description_dir.glob("*.txt"), key=numeric_sort_key)
    limit = args.limit if args.limit is not None else MAX_DESCRIPTIONS
    if limit is not None and limit > 0:
        files = files[:limit]
    return files


def output_path_for(description_file: Path, args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    return Path(args.output_dir) / f"{description_file.stem}_vgdl_phi4mini_grpo.txt"


def validate_adapter_path(adapter_path: Path) -> None:
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [name for name in required_files if not (adapter_path / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Adapter directory '{adapter_path}' is missing: {missing_text}. "
            "Use a GRPO checkpoint/last-model directory, not the zero-shot folder."
        )


def print_adapter_summary(adapter_path: Path) -> None:
    print(f"Adapter path: {adapter_path}")
    adapter_file = adapter_path / "adapter_model.safetensors"
    print(f"Adapter weights: {adapter_file.name} ({adapter_file.stat().st_size / (1024 * 1024):.1f} MB)")

    trainer_state_path = adapter_path / "trainer_state.json"
    if trainer_state_path.exists():
        try:
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                trainer_state = json.load(f)
            print(
                "Training state: "
                f"global_step={trainer_state.get('global_step')}, "
                f"epoch={trainer_state.get('epoch')}"
            )
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Training state: unreadable ({exc})")


def clean_output(text: str) -> str:
    text = text.strip()
    text = text.replace("BasicGameSpriteSet", "BasicGame\n    SpriteSet", 1)
    text = text.replace(
        "BasicGame\n    SpriteSet\nBasicGame\n    SpriteSet",
        "BasicGame\n    SpriteSet",
        1,
    )
    text = text.replace(
        "BasicGame\n    SpriteSet\n    SpriteSet",
        "BasicGame\n    SpriteSet",
        1,
    )
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    if text.startswith("BasicGameBasicGame"):
        text = text.replace("BasicGameBasicGame", "BasicGame", 1)
    start = text.find("BasicGame")
    if start > 0:
        text = text[start:]
    for marker in (
        "\n```",
        "\nNote:",
        "\nReview ",
        "\nExplanation:",
        "\nThis ",
        "\nEndgameState",
        "\nFunction(",
        "\nSuccessive",
        "\nFailed",
        "\nLostLife",
    ):
        marker_pos = text.find(marker)
        if marker_pos != -1:
            text = text[:marker_pos]
    if not text.startswith("BasicGame"):
        first_section = min(
            [pos for pos in (text.find("SpriteSet"), text.find("LevelMapping")) if pos >= 0],
            default=-1,
        )
        if first_section >= 0:
            text = "BasicGame\n" + text[first_section:].lstrip()
    return text.strip()


def load_model(args: argparse.Namespace):
    adapter_path = Path(args.adapter_path)
    validate_adapter_path(adapter_path)
    print_adapter_summary(adapter_path)

    print(f"Loading base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": args.attn_implementation,
        "max_memory": {0: args.gpu_max_memory, "cpu": args.cpu_max_memory},
    }

    if args.no_4bit:
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **kwargs)

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")
    active_adapters = getattr(model, "active_adapters", None)
    if callable(active_adapters):
        active_adapters = active_adapters()
    print(f"Active PEFT adapter(s): {active_adapters}")

    if args.merge_adapter:
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_vgdl(
    model,
    tokenizer,
    description: str,
    max_new_tokens: int,
    prompt_mode: str,
    adapter_enabled: bool = True,
) -> str:
    prompt_text = build_prompt(description)
    if prompt_mode == "chat":
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = prompt_text

    inputs = tokenizer(prompt, return_tensors="pt")
    input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    print(f"Input tokens: {inputs['input_ids'].shape[-1]}")
    start = time.time()
    adapter_context = nullcontext()
    if not adapter_enabled and hasattr(model, "disable_adapter"):
        adapter_context = model.disable_adapter()

    if adapter_enabled:
        print("Generation mode: base model + GRPO LoRA adapter")
    else:
        print("Generation mode: base model only (LoRA adapter disabled)")

    with adapter_context, torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    print(f"Generated in {elapsed:.2f}s")

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_single(description: str, label: str, args: argparse.Namespace, model, tokenizer) -> str:
    print(f"\n--- {label} ---")
    if args.compare_base:
        base_output = generate_vgdl(
            model=model,
            tokenizer=tokenizer,
            description=description,
            max_new_tokens=args.max_new_tokens,
            prompt_mode=args.prompt_mode,
            adapter_enabled=False,
        )
        print("\n=== Base Model Output (adapter disabled) ===")
        print(clean_output(base_output))

    raw_output = generate_vgdl(
        model=model,
        tokenizer=tokenizer,
        description=description,
        max_new_tokens=args.max_new_tokens,
        prompt_mode=args.prompt_mode,
        adapter_enabled=not args.disable_adapter,
    )
    vgdl = clean_output(raw_output)
    print("\n=== Generated VGDL ===")
    print(vgdl)
    return vgdl


def main() -> None:
    args = parse_args()
    if args.merge_adapter and (args.disable_adapter or args.compare_base):
        raise ValueError(
            "--merge-adapter cannot be combined with --disable-adapter or --compare-base."
        )

    # ---- Inline description (highest priority) ----
    if args.description_text:
        if args.dry_run:
            print(build_prompt(args.description_text))
            return
        print(f"Loading base model ({BASE_MODEL_NAME}) + adapter ({args.adapter_path})...")
        t0 = time.time()
        model, tokenizer = load_model(args)
        print(f"Model loaded in {time.time() - t0:.2f}s")
        vgdl = run_single(args.description_text, "inline description", args, model, tokenizer)
        if args.output_path:
            os.makedirs(Path(args.output_path).parent, exist_ok=True)
            with open(args.output_path, "w", encoding="utf-8") as f:
                f.write(vgdl)
            print(f"\nSaved to {args.output_path}")
        return

    # ---- Single description file ----
    if args.description_file:
        if args.dry_run:
            description = read_description(args.description_file)
            print(build_prompt(description))
            return
        print(f"Loading base model ({BASE_MODEL_NAME}) + adapter ({args.adapter_path})...")
        t0 = time.time()
        model, tokenizer = load_model(args)
        print(f"Model loaded in {time.time() - t0:.2f}s")
        description = read_description(args.description_file)
        vgdl = run_single(description, args.description_file, args, model, tokenizer)
        out = Path(args.output_path) if args.output_path else (
            Path(args.output_dir) / f"{Path(args.description_file).stem}_vgdl_phi4mini_grpo.txt"
        )
        os.makedirs(out.parent, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(vgdl)
        print(f"\nSaved to {out}")
        return

    # ---- Batch mode ----
    description_files = select_description_files(args)
    if not description_files:
        raise RuntimeError("No description files found. Check --description-dir or --limit.")

    if args.dry_run:
        effective_limit = args.limit if args.limit is not None else MAX_DESCRIPTIONS
        print(f"Adapter:         {args.adapter_path}")
        print(f"Description dir: {args.description_dir}")
        print(f"Output dir:      {args.output_dir}")
        print(f"Selected limit:  {effective_limit}")
        print("Selected description files:")
        for p in description_files:
            print(f"  - {p}")
        first_description = read_description(str(description_files[0]))
        print("\nFirst prompt preview:\n")
        print(build_prompt(first_description))
        return

    print(f"Loading base model ({BASE_MODEL_NAME}) + adapter ({args.adapter_path})...")
    t0 = time.time()
    model, tokenizer = load_model(args)
    print(f"Model loaded in {time.time() - t0:.2f}s")

    for index, description_file in enumerate(description_files, 1):
        output_path = output_path_for(description_file, args)
        print(f"\n[{index}/{len(description_files)}] {description_file} -> {output_path}")
        description = read_description(str(description_file))
        raw_output = generate_vgdl(
            model=model,
            tokenizer=tokenizer,
            description=description,
            max_new_tokens=args.max_new_tokens,
            prompt_mode=args.prompt_mode,
            adapter_enabled=not args.disable_adapter,
        )
        vgdl = clean_output(raw_output)
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(vgdl)
        print(f"Output saved to {output_path}")
        print(vgdl)


if __name__ == "__main__":
    main()
