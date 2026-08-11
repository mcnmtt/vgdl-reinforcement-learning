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

        if in_term:
            tokens = stripped.split()
            if not tokens:
                continue

            condition = tokens[0]
            if condition not in {
                "SpriteCounter",
                "MultiSpriteCounter",
                "Timeout",
            }:
                continue

            parameters = {}
            for token in tokens[1:]:
                if "=" in token:
                    key, value = token.split("=", 1)
                    parameters[key] = value

            if condition == "SpriteCounter":
                relevant_keys = ("stype", "limit", "win")
            elif condition == "MultiSpriteCounter":
                relevant_keys = ("stype1", "stype2", "limit", "win")
            else:
                relevant_keys = ("limit", "win", "count_score")

            signature = tuple(
                (key, parameters[key])
                for key in relevant_keys
                if key in parameters
            )
            terms.add((condition, signature))

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
BasicGame no_players=2 square_size=30
    SpriteSet
        floor > Immovable img=oryx/backGrey hidden=True

        bullet > Missile shrinkfactor=0.5
            bulletA > singleton=True orientation=RIGHT
                rockAb > img=oryx/planet1
                    rockAb0 > speed=0.5
                    rockAb1 > speed=0.3
                paperAb > img=oryx/scroll1
                    paperAb0 > speed=0.5
                    paperAb1 > speed=0.3
                scissorsAb > img=oryx/axe2
                    scissorsAb0 > speed=0.5
                    scissorsAb1 > speed=0.3
                lizardAb > img=oryx/dragon1
                    lizardAb0 > speed=0.5
                    lizardAb1 > speed=0.3
                spockAb > img=oryx/druid1
                    spockAb0 > speed=0.5
                    spockAb1 > speed=0.3
            bulletB > singleton=True orientation=LEFT
                rockBb > img=oryx/planet1
                    rockBb0 > speed=0.5
                    rockBb1 > speed=0.3
                paperBb > img=oryx/scroll1
                    paperBb0 > speed=0.5
                    paperBb1 > speed=0.3
                scissorsBb > img=oryx/axe2
                    scissorsBb0 > speed=0.5
                    scissorsBb1 > speed=0.3
                lizardBb > img=oryx/dragon1
                    lizardBb0 > speed=0.5
                    lizardBb1 > speed=0.3
                spockBb > img=oryx/druid1
                    spockBb0 > speed=0.5
                    spockBb1 > speed=0.3

        choice > Flicker shrinkfactor=0.5
            choiceA > singleton=True hidden=False,True invisible=False,True
                rockA > img=oryx/planet1
                paperA > img=oryx/scroll1
                scissorsA > img=oryx/axe2
                lizardA > img=oryx/dragon1
                spockA > img=oryx/druid1
            choiceB > singleton=True hidden=True,False invisible=True,False
                rockB > img=oryx/planet1
                paperB > img=oryx/scroll1
                scissorsB > img=oryx/axe2
                lizardB > img=oryx/dragon1
                spockB > img=oryx/druid1

        avatar > #frameRate=16
            avatarA > ShootOnlyAvatar stype=rockA,paperA,scissorsA img=newset/bandit1 orientation=RIGHT
            avatarB > ShootOnlyAvatar stype=rockB,paperB,scissorsB img=newset/bandit1h orientation=LEFT
            avatarA5 > ShootOnlyAvatar stype=rockA,paperA,scissorsA,lizardA,spockA img=newset/bandit1 orientation=RIGHT
            avatarB5 > ShootOnlyAvatar stype=rockB,paperB,scissorsB,lizardA,spockA img=newset/bandit1h orientation=LEFT

        buzzer > Immovable shrinkfactor=0.5
            buzzerA > hidden=False,True invisible=False,True
                buzzerRockA > img=oryx/planet1
                buzzerPaperA > img=oryx/scroll1
                buzzerScissorsA > img=oryx/axe2
                buzzerLizardA > img=oryx/dragon1
                buzzerSpockA > img=oryx/druid1
            buzzerB > hidden=True,False invisible=True,False
                buzzerRockB > img=oryx/planet1
                buzzerPaperB > img=oryx/scroll1
                buzzerScissorsB > img=oryx/axe2
                buzzerLizardB > img=oryx/dragon1
                buzzerSpockB > img=oryx/druid1

        timer >
            timer1 > Immovable invisible=True hidden=True
                timer10 >
                timer11 >
            timer2 > Immovable invisible=True hidden=True
                timer20 >
                timer21 >
        choose > Immovable invisible=True hidden=True
            choose0 >
            choose1 >

        wall > Immovable img=oryx/wall3 autotiling=True


    LevelMapping
        . > floor
        A > avatarA floor
        B > avatarB floor

        C > avatarA5 floor
        D > avatarB5 floor

        0 > timer10 floor buzzerRockA
        1 > timer10 floor buzzerRockB

        2 > timer20 floor buzzerRockA
        3 > timer20 floor buzzerRockB

        4 > timer11 floor buzzerRockA
        5 > timer11 floor buzzerRockB

        6 > timer21 floor buzzerRockA
        7 > timer21 floor buzzerRockB

        w > floor wall

    InteractionSet
        timer10 TIME > spawn stype=choose0 timer=50 repeating=True
        timer20 TIME > spawn stype=choose0 timer=20 repeating=True
        timer11 TIME > spawn stype=choose1 timer=50 repeating=True
        timer21 TIME > spawn stype=choose1 timer=20 repeating=True

        bullet wall > killSprite

        buzzerA rockA > transformTo stype=buzzerRockA killSecond=True
        buzzerA paperA > transformTo stype=buzzerPaperA killSecond=True
        buzzerA scissorsA > transformTo stype=buzzerScissorsA killSecond=True
        buzzerA lizardA > transformTo stype=buzzerLizardA killSecond=True
        buzzerA spockA > transformTo stype=buzzerSpockA killSecond=True

        buzzerB rockB > transformTo stype=buzzerRockB killSecond=True
        buzzerB paperB > transformTo stype=buzzerPaperB killSecond=True
        buzzerB scissorsB > transformTo stype=buzzerScissorsB killSecond=True
        buzzerB lizardB > transformTo stype=buzzerLizardB killSecond=True
        buzzerB spockB > transformTo stype=buzzerSpockB killSecond=True

        choose0 buzzerRockA > spawn stype=rockAb0
        choose0 buzzerPaperA > spawn stype=paperAb0
        choose0 buzzerScissorsA > spawn stype=scissorsAb0
        choose0 buzzerLizardA > spawn stype=lizardAb0
        choose0 buzzerSpockA > spawn stype=spockAb0

        choose0 buzzerRockB > spawn stype=rockBb0
        choose0 buzzerPaperB > spawn stype=paperBb0
        choose0 buzzerScissorsB > spawn stype=scissorsBb0
        choose0 buzzerLizardB > spawn stype=lizardBb0
        choose0 buzzerSpockB > spawn stype=spockBb0

        choose1 buzzerRockA > spawn stype=rockAb1
        choose1 buzzerPaperA > spawn stype=paperAb1
        choose1 buzzerScissorsA > spawn stype=scissorsAb1
        choose1 buzzerLizardA > spawn stype=lizardAb1
        choose1 buzzerSpockA > spawn stype=spockAb1

        choose1 buzzerRockB > spawn stype=rockBb1
        choose1 buzzerPaperB > spawn stype=paperBb1
        choose1 buzzerScissorsB > spawn stype=scissorsBb1
        choose1 buzzerLizardB > spawn stype=lizardBb1
        choose1 buzzerSpockB > spawn stype=spockBb1

        choose buzzer > killSprite

        rockAb rockBb > killBoth
        paperAb paperBb > killBoth
        scissorsAb scissorsBb > killBoth
        lizardAb lizardBb > killBoth
        spockAb spockBb > killBoth


        rockAb paperBb > killSprite #paper beats rock
        rockBb paperAb > killSprite

        spockAb paperBb > killSprite #paper beats spock
        spockBb paperAb > killSprite

        paperAb scissorsBb > killSprite #scissors beats paper
        paperBb scissorsAb > killSprite

        lizardAb scissorsBb > killSprite #scissors beats lizard
        lizardBb scissorsAb > killSprite

        scissorsAb rockBb > killSprite #rock beats scissors
        scissorsBb rockAb > killSprite

        lizardAb rockBb > killSprite #rock beats lizard
        lizardBb rockAb > killSprite

        rockAb spockBb > killSprite #spock beats rock
        rockBb spockAb > killSprite

        scissorsAb spockBb > killSprite #spock beats scissors
        scissorsBb spockAb > killSprite

        paperAb lizardAb > killSprite #lizard beats paper
        paperBb lizardBb > killSprite

        spockAb lizardAb > killSprite #lizard beats spock
        spockBb lizardBb > killSprite


        bulletB avatarA > killSprite scoreChange=0,1
        bulletA avatarB > killSprite scoreChange=1,0

    TerminationSet
        Timeout limit=500 count_score=True
"""

    generated_vgdl = """
BasicGame square_size=30 key_handler=Pulse
    SpriteSet

        floor > Immovable img=oryx/backGrey hidden=True
        avatar  > ShootAvatar stype=projectile frameRate=8
            banditA > orientation=RIGHT img=newset/bandit1
            banditB > orientation=LEFT img=newset/bandit2
        wall > Immovable color=LIGHTGRAY autotiling=true img=oryx/wall6
        projectile > MissileShrinker singleton=True
            rock   > orientation=RIGHT speed=0.4 img=oryx/planet shrinkfactor=0.7
            paper  > orientation=RIGHT speed=0.4 img=oryx/scroll shrinkfactor=0.7
            scissors > orientation=RIGHT speed=0.4 img=oryx/axe shrinkfactor=0.7
            lizard > orientation=RIGHT speed=0.4 img=oryx/dragon shrinkfactor=0.7
            spock  > orientation=RIGHT speed=0.4 img=oryx/druid shrinkfactor=0.7
        choice > Flicker color=WHITE hidden=True
            choiceA >
                rockChoiceA > img=oryx/planet
                paperChoiceA > img=oryx/scroll
                scissorsChoiceA > img=oryx/axe
                lizardChoiceA > img=oryx/dragon
                spockChoiceA > img=oryx/druid
            choiceB >
                rockChoiceB > img=oryx/planet
                paperChoiceB > img=oryx/scroll
                scissorsChoiceB > img=oryx/axe
                lizardChoiceB > img=oryx/dragon
                spockChoiceB > img=oryx/druid
        buzzer > Passive
            buzzerA >
                rock
"""

    result = vgdl_similarity(target_vgdl, generated_vgdl)

    print("Sprite similarity:", result["sprite_similarity"])
    print("Interaction similarity:", result["interaction_similarity"])
    print("Termination similarity:", result["termination_similarity"])
    print("Final similarity:", result["final_score"])


# ==============================

if __name__ == "__main__":
    main()
