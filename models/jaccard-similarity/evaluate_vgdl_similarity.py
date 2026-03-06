@ -1,268 +0,0 @@
# ==============================
# Configurazione
# ==============================

ABSTRACT_SPRITES = {"structure", "moving"}

INTERACTION_MAP = {
    "stepBack": "block",
    "bounceForward": "block",
    "wallStop": "block",
    "killSprite": "kill",
    "killIfAlive": "kill",
}

SYMMETRIC_INTERACTIONS = {"kill", "block"}


# ==============================
# Utility
# ==============================

def normalize_interaction_name(name):
    return INTERACTION_MAP.get(name, name)


def canonical_interaction(s1, interaction, s2):

    interaction = normalize_interaction_name(interaction)

    if interaction in SYMMETRIC_INTERACTIONS:
        a, b = sorted([s1, s2])
        return (a, interaction, b)

    return (s1, interaction, s2)


# ==============================
# Parsing SpriteSet
# ==============================

def extract_leaf_sprites(vgdl):

    sprites = set()
    lines = vgdl.splitlines()

    in_sprite = False

    for raw_line in lines:

        stripped = raw_line.strip()

        if stripped.startswith("SpriteSet"):
            in_sprite = True
            continue

        if in_sprite and stripped.startswith(("InteractionSet", "TerminationSet", "LevelMapping")):
            in_sprite = False
            continue

        if in_sprite and ">" in stripped:

            name = stripped.split(">")[0].strip()

            if name and name not in ABSTRACT_SPRITES:
                sprites.add(name)

    return sprites


# ==============================
# Parsing InteractionSet
# ==============================

def extract_interactions(vgdl):

    interactions = set()
    lines = vgdl.splitlines()

    in_interaction = False

    for raw_line in lines:

        stripped = raw_line.strip()

        if stripped.startswith("InteractionSet"):
            in_interaction = True
            continue

        if in_interaction and stripped.startswith(("SpriteSet", "TerminationSet", "LevelMapping")):
            in_interaction = False
            continue

        if in_interaction and ">" in stripped:

            left, right = stripped.split(">", 1)

            tokens = left.strip().split()

            if len(tokens) == 2:

                s1, s2 = tokens
                interaction = right.strip().split()[0]

                if s1 not in ABSTRACT_SPRITES and s2 not in ABSTRACT_SPRITES:

                    interactions.add(
                        canonical_interaction(s1, interaction, s2)
                    )

    return interactions


# ==============================
# Parsing TerminationSet
# ==============================

def extract_termination(vgdl):

    terms = set()
    lines = vgdl.splitlines()

    in_term = False

    for raw_line in lines:

        stripped = raw_line.strip()

        if stripped.startswith("TerminationSet"):
            in_term = True
            continue

        if in_term and stripped.startswith(("SpriteSet", "InteractionSet", "LevelMapping")):
            in_term = False
            continue

        if in_term and stripped.startswith("SpriteCounter"):

            stype = None
            win = None

            for token in stripped.split():

                if token.startswith("stype="):
                    stype = token.split("=")[1]

                if token.startswith("win="):
                    win = token.split("=")[1]

            if stype:
                terms.add((stype, win))

    return terms


# ==============================
# Similarity
# ==============================

def jaccard(a, b):

    union = len(a | b)

    if union == 0:
        return 1.0

    return len(a & b) / union


def vgdl_similarity(vgdl1, vgdl2):

    sprites1 = extract_leaf_sprites(vgdl1)
    sprites2 = extract_leaf_sprites(vgdl2)

    inter1 = extract_interactions(vgdl1)
    inter2 = extract_interactions(vgdl2)

    term1 = extract_termination(vgdl1)
    term2 = extract_termination(vgdl2)

    sprite_sim = jaccard(sprites1, sprites2)
    interaction_sim = jaccard(inter1, inter2)
    termination_sim = jaccard(term1, term2)

    final_score = (
        0.25 * sprite_sim +
        0.50 * interaction_sim +
        0.25 * termination_sim
    )

    return {
        "sprite_similarity": sprite_sim,
        "interaction_similarity": interaction_sim,
        "termination_similarity": termination_sim,
        "final_score": final_score
    }


# ==============================
# MAIN
# ==============================

def main():

    target_vgdl = """
BasicGame
    SpriteSet    
        pad    > Immovable color=BLUE 
        avatar > AimedAvatar stype=bullet
        bullet > Missile physicstype=GravityPhysics speed=25 singleton=True
            
    TerminationSet
        SpriteCounter stype=pad    win=True     
        SpriteCounter stype=avatar win=False     
           
    InteractionSet
        wall bullet > killSprite 
        bullet wall > killSprite 
        pad bullet > killSprite
        avatar wall > stepBack
        avatar EOS > stepBack
        bullet EOS > killSprite

    LevelMapping
        G > pad
"""

    generated_vgdl = """
BasicGame

    SpriteSet
        structure > Immovable
            wall > color=BLACK
            pad  > color=GREEN

        moving > physicstype=GravityPhysics
            avatar > ShootAvatar stype=bullet color=BLUE
            bullet > Missile color=RED singleton=True

    InteractionSet
        avatar wall   > bounceForward
        avatar EOS    > bounceForward

        bullet wall   > killSprite
        bullet pad    > killSprite
        bullet EOS    > killSprite

    TerminationSet
        SpriteCounter stype=pad    win=True  limit=0
        SpriteCounter stype=avatar win=False limit=0

    LevelMapping
        A > avatar
        w > wall
        p > pad
"""

    result = vgdl_similarity(target_vgdl, generated_vgdl)

    print("Sprite similarity:", result["sprite_similarity"])
    print("Interaction similarity:", result["interaction_similarity"])
    print("Termination similarity:", result["termination_similarity"])
    print("Final similarity:", result["final_score"])


# ==============================

if __name__ == "__main__":
    main()