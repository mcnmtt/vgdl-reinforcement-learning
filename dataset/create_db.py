import os
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ======================
# Configuration
# ======================
VGDL_FOLDER = "vgdl_files"
DESC_FOLDER = "descriptions"
os.makedirs(DESC_FOLDER, exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a game designer describing a video game to someone who will implement it. Given a VGDL game specification, generate a structured natural language description of the game written as a human who knows the game rules thoroughly but has no knowledge of VGDL or any programming language.
Follow these rules:
Describe behaviors and roles in plain English, never use VGDL class names, effect names, or technical keywords.
Instead of class names, describe what the sprite does: e.g. instead of "Walker", write "an enemy that moves horizontally and bounces off walls".
Instead of effect names, describe what happens: e.g. instead of "killIfFromAbove", write "is destroyed only if the other sprite arrives from above".
Instead of "EOS", write "the edge of the screen".
When a sprite belongs to a group that shares physics or behavior, describe the shared behavior in plain English: e.g. "all enemies and the avatar are affected by gravity".
When an interaction has parameters, describe their meaning: e.g. instead of "friction=0.1", write "slows down slightly upon contact".
When a sprite has parameters, describe their effect: e.g. instead of "orientation=UP speed=0.1", write "moves upward slowly".
Always use the following fixed structure:
Player role:
[Describe what the player controls and how it moves, including any special physics like gravity or inertia.]
Entities:
[List every object in the game. For each, describe its appearance, behavior, and role without using technical names.]
Interactions:
[List every interaction as a plain English sentence describing what happens when two objects meet.]
Win condition:
[Describe in plain English when the player wins.]
Lose condition:
[Describe in plain English when the player loses.]
Objective:
[One or two sentences summarizing the goal from the player's perspective.]
Here is the VGDL to describe:"""

# ======================
# Get sorted list of VGDL files
# ======================
all_files = sorted(
    [f for f in os.listdir(VGDL_FOLDER) if f.endswith(".txt")],
    key=lambda x: int(x.split("_")[0])
)

print(f"Found {len(all_files)} VGDL files.\n")

# ======================
# Generate descriptions
# ======================
for i, filename in enumerate(all_files):
    vgdl_path = os.path.join(VGDL_FOLDER, filename)
    desc_path = os.path.join(DESC_FOLDER, filename)

    # Skip if description already exists
    if os.path.exists(desc_path):
        print(f"[{i+1}/{len(all_files)}] Skipping (already exists): {filename}")
        continue

    # Read VGDL file
    with open(vgdl_path, "r") as f:
        vgdl_content = f.read()

    print(f"[{i+1}/{len(all_files)}] Generating description for: {filename}")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\n{vgdl_content}"
                }
            ]
        )

        description = response.content[0].text.strip()

        # Save description
        with open(desc_path, "w") as f:
            f.write(description)

        print(f"    → Saved to {desc_path}")

        # Avoid hitting rate limits
        time.sleep(0.5)

    except Exception as e:
        print(f"    ERROR on {filename}: {e}")
        time.sleep(5)  # Wait longer on error before retrying

print("\nDone! All descriptions generated.")