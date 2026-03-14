import torch
import time
from transformers import AutoProcessor, AutoModelForImageTextToText

# ======================
# GPU Configuration
# ======================
if not torch.cuda.is_available():
    print("Warning: CUDA not available. Falling back to CPU.")
else:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

# ======================
# Model Loading
# ======================
print("Loading model... (this may take 1-2 minutes)")
start = time.time()

model_name = "Qwen/Qwen3.5-4B"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    low_cpu_mem_usage=True,
)
model.eval()

end = time.time()
print(f"Model loaded in {round(end-start,2)} seconds\n")

# ======================
# Game Description Input
# ======================
game_description = """
Player role:
The player controls an avatar that can rotate its aim direction and shoot a projectile. The avatar cannot move around the level freely — it can only adjust its aiming angle. It cannot pass through walls or the edge of the screen and is pushed back if it tries.
Entities:

The avatar is the player-controlled character. It stays in place but rotates to aim, and fires a projectile when the player shoots.
The bullet is a fast-moving projectile fired by the avatar. It is affected by gravity, so it follows a curved trajectory. Only one bullet can exist at a time — a new one cannot be fired until the previous one disappears.
The pad is a blue static target object placed in the level. It does not move.
Walls are static objects that block the avatar and destroy bullets on contact.

Interactions:

When a bullet hits a wall from either direction, the bullet is destroyed.
When a bullet hits a pad, the pad is destroyed.
When the avatar tries to move into a wall, it is pushed back to its previous position.
When the avatar reaches the edge of the screen, it is pushed back to its previous position.
When a bullet reaches the edge of the screen, it is destroyed.

Win condition:
The player wins when all blue pad objects have been destroyed.
Lose condition:
The player loses if the avatar is destroyed.
Objective:
Carefully aim and fire the gravity-affected projectile to destroy all blue pads in the level, taking into account the bullet's curved trajectory caused by gravity.
"""

# ======================
# Few-shot Prompt (Description → VGDL)
# ======================
print("Preparing prompt...")

few_shot_example = """
Example 1 — Basic sprites and interactions:
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

Example 2 — Child sprites (shooter + projectile):
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

Example 3 — Sprite groups and EOS:
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

Example 4 — Nested sprite groups:
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
"""

prompt = f"""You are an expert in VGDL (Video Game Description Language).
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
- MovingAvatar             → 4-directional player avatar
- HorizontalAvatar         → player avatar that moves left/right only
- VerticalAvatar           → player avatar that moves up/down only
- OrientedAvatar           → player avatar that retains orientation
- FlakAvatar               → horizontal avatar that shoots upward (child sprite required)
- ShootAvatar              → oriented avatar that shoots forward (child sprite required)
- AimedAvatar              → player avatar that can change the direction of firing, but not move
- AimedFlakAvatar          → horizontal avatar with aim control and left/right movement (child sprite required)
- InertialAvatar           → avatar with continuous inertial physics
- MarioAvatar              → avatar with gravity and jump mechanics
- RotatingAvatar           → avatar that rotates and moves forward/backward relative to its orientation
- RotatingFlippingAvatar   → like RotatingAvatar but DOWN rotates 180°
- NoisyRotatingFlippingAvatar → RotatingFlippingAvatar with stochastic noise (noiseLevel=0.1)
- Immovable                → static object, cannot be moved (walls, goals, holes)
- Passive                  → object that can be pushed or interacted with
- ResourcePack             → collectible resource object
- Flicker                  → sprite that disappears after a few timesteps
- OrientedFlicker          → short-lived directional sprite (e.g. sword slash)
- Missile                  → moves continuously in a fixed direction
- RandomMissile            → missile with randomized direction and speed at initialization
- ErraticMissile           → missile that randomly changes direction with probability prob=
- Bomber                   → missile that also spawns sprites periodically (requires stype=)
- Walker                   → moves horizontally, bounces off walls
- WalkJumper               → moves horizontally and occasionally jumps
- RandomNPC                → NPC that moves randomly each step
- RandomInertial           → oriented sprite with continuous physics that moves randomly
- Chaser                   → NPC that moves toward a target sprite type (requires stype=)
- Fleeing                  → NPC that moves away from a target sprite type (requires stype=)
- AStarChaser              → NPC that uses A* search to chase a target sprite type (requires stype=)
- Spreader                 → spreads to adjacent cells over time
- Conveyor                 → static object that moves other sprites along its orientation
- Portal                   → teleports sprites that touch it (requires stype=)
- SpawnPoint               → spawns sprites of a given type over time (requires stype=)

VALID INTERACTION EFFECTS (use ONLY these):
- killSprite               → destroy the first sprite
- cloneSprite              → clone the first sprite
- transformTo              → replace first sprite with another type (MUST include stype=ClassName)
- stepBack                 → undo the move of the first sprite
- undoAll                  → undo last move of all sprites
- bounceForward            → push first sprite in the partner's last direction
- bounceDirection          → bounce first sprite based on center-to-center direction
- conveySprite             → move first sprite along partner's orientation
- windGust                 → like conveySprite but with stochastic force variation
- slipForward              → stochastically move first sprite forward along its orientation
- attractGaze              → stochastically rotate first sprite's orientation toward partner's
- turnAround               → make first sprite reverse direction (stepBack + reverseDirection)
- reverseDirection         → reverse the orientation of the first sprite
- flipDirection            → set orientation of first sprite randomly
- wallBounce               → bounce first sprite off wall orthogonally
- wallStop                 → stop first sprite at wall, slide along it
- killIfSlow               → kill first sprite if relative speed is below threshold
- killIfFromAbove          → kill first sprite only if partner arrived from above
- killIfAlive              → kill first sprite only if partner is not already in kill list
- killIfHasMore            → kill first sprite if it has more than limit of a resource (requires resource= limit=)
- killIfHasLess            → kill first sprite if it has less than limit of a resource (requires resource= limit=)
- killIfOtherHasMore       → kill first sprite if partner has more than limit of a resource (requires resource= limit=)
- killIfOtherHasLess       → kill first sprite if partner has less than limit of a resource (requires resource= limit=)
- wrapAround               → wrap first sprite to opposite side of screen
- pullWithIt               → carry first sprite along with partner's movement
- teleportToExit           → teleport first sprite to a portal/exit
- collectResource          → collect a resource pack into partner's inventory
- changeResource           → increment a specific resource in first sprite (requires resource= value=)
- spawnIfHasMore           → spawn a sprite if first sprite has more than limit of a resource (requires resource= stype= limit=)

VALID TERMINATION CONDITIONS (use ONLY these):
- SpriteCounter stype=<n> limit=<n> win=True/False
- MultiSpriteCounter stype1=<n> stype2=<n> limit=<n> win=True/False
- Timeout limit=<n> win=True/False   (limit must be > 0)

EDGE OF SCREEN — CRITICAL RULE:
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

{few_shot_example}

Now convert this description:
Description:
{game_description.strip()}

Respond with ONLY the raw VGDL code block, starting with 'BasicGame'. \
No explanation, no markdown fences, no comments, no reasoning.

VGDL:
BasicGame"""

# ======================
# Tokenization
# ======================
print("Tokenizing input...")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}
        ]
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking = False,
).to(model.device)
print(f"Input tokens: {inputs['input_ids'].shape[-1]}\n")

# ======================
# Generation
# ======================
print("Generating VGDL...\n")

start = time.time()
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
end = time.time()

# ======================
# Extract ONLY new tokens (the generated VGDL)
# ======================
input_length = inputs["input_ids"].shape[-1]
generated_ids = output_ids[0][input_length:]
raw_output = processor.decode(generated_ids, skip_special_tokens=True)

# ======================
# Save raw output to file
# ======================
output_path = "models/qwen3.5/results/vgdl/gioco_vgdl_qwen.txt"
with open(output_path, "w") as f:
    f.write(raw_output)
print(f"\nOutput saved to {output_path}")