import sys
import os

# Aggiunge py-vgdl al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py-vgdl'))

from vgdl.core import VGDLParser # type: ignore


def validate_vgdl(game_file, level_file=None):

    result = {
        "valid": True,
        "errors": []
    }

    try:
        with open(game_file, "r") as f:
            game_str = f.read()

        parser = VGDLParser()
        game = parser.parseGame(game_str)
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Parsing error: {e}")
        return result

    try:
        sprites = set(game.sprite_constr.keys())
    except Exception:
        sprites = set()

    # controlla interazioni
    try:
        valid_sprites = set()
        special_sprites = {"EOS", "wall"}

        # sprite concreti
        valid_sprites.update(game.sprite_constr.keys())

        # sprite astratti dalla gerarchia
        for _, (_, _, stypes) in game.sprite_constr.items():
            valid_sprites.update(stypes)

        for interaction in game.collision_eff:
            sprite1 = interaction[0]
            sprite2 = interaction[1]

            if sprite1 not in valid_sprites and sprite1 not in special_sprites:
                result["errors"].append(
                    f"Interaction uses undefined sprite: {sprite1}"
                )

            if sprite2 not in valid_sprites and sprite2 not in special_sprites:
                result["errors"].append(
                    f"Interaction uses undefined sprite: {sprite2}"
                )
    except Exception:
        pass

    # controlla termination
    try:
        for term in game.terminations:
            if hasattr(term, "stype"):
                if term.stype not in sprites:
                    result["errors"].append(
                        f"Termination references undefined sprite: {term.stype}"
                    )
    except Exception:
        pass

    # prova a costruire il livello
    if level_file:
        try:
            game.buildLevel(level_file)
        except Exception as e:
            result["errors"].append(f"Level build error: {e}")

    if result["errors"]:
        result["valid"] = False

    return result


if __name__ == "__main__":

    import sys

    game_file = sys.argv[1]
    level_file = sys.argv[2] if len(sys.argv) > 2 else None

    report = validate_vgdl(game_file, level_file)

    if report["valid"]:
        print("VGDL VALID")
    else:
        print("VGDL INVALID")

    for e in report["errors"]:
        print(("-", e))