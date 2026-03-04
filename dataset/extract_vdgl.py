import os
import re

# Cartelle di input e output
INPUT_FOLDER = "dataset/vgdl_files_row"
OUTPUT_FOLDER = "dataset/vgdl_files"

# Regex per estrarre il blocco VGDL (stringa multilinea che contiene BasicGame)
VGDL_REGEX = re.compile(
    r'("""|\'\'\')\s*(BasicGame.*?)(?:\1)',
    re.DOTALL
)

def extract_vgdl_from_file(py_path):
    """Estrae la sezione BasicGame da un file .py."""
    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = VGDL_REGEX.search(content)
    if not match:
        return None

    return match.group(2).strip()


def process_files():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Ordina i file per avere ID coerenti
    py_files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.endswith(".py"))

    for idx, filename in enumerate(py_files, start=1):
        py_path = os.path.join(INPUT_FOLDER, filename)
        vgdl_text = extract_vgdl_from_file(py_path)

        if vgdl_text is None:
            print(f"[SKIP] Nessun blocco VGDL in: {filename}")
            continue

        base_name = filename.replace(".py", "")
        txt_filename = f"{idx}_{base_name}.txt"
        txt_path = os.path.join(OUTPUT_FOLDER, txt_filename)

        with open(txt_path, "w", encoding="utf-8") as out:
            out.write(vgdl_text)

        print(f"[OK] Estratto: {txt_filename}")


if __name__ == "__main__":
    process_files()
