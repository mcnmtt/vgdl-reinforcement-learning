import os
import glob
import datetime
from dotenv import load_dotenv

from openai import OpenAI
from pymongo import MongoClient

# Carica variabili da .env
load_dotenv()

# =======================
# CONFIGURAZIONE
# =======================

# Cartella che contiene i file VGDL (es. .txt, .vgdl, ecc.)
VGDL_FOLDER = r"dataset\vgdl_files"

# Pattern dei file da leggere
FILE_PATTERN = "*.txt"

# Modello OpenAI da usare
OPENAI_MODEL = "gpt-5.1"  

# Prompt di sistema fisso
SYSTEM_PROMPT = (
    "Sei un convertitore neutrale da codice VGDL a descrizioni testuali. "
    "Ti fornirò del codice VGDL. Il tuo compito è generare una DESCRIZIONE TESTUALE "
    "OGGETTIVA, COMPLETA E STANDARDIZZATA del gioco definito da quel codice, senza "
    "aggiungere elementi inventati o modificare la logica. "
    "La descrizione deve includere, nell’ordine: "
    "1. Panoramica del gioco: quali sprite esistono e qual è il loro ruolo generale. "
    "2. Descrizione dettagliata di ogni SpriteSet: proprietà, comportamenti, colori, "
    "orientamenti, fisica, gerarchie. "
    "3. Descrizione dettagliata dell’InteractionSet: cosa succede quando due sprite "
    "entrano in contatto. "
    "4. Descrizione del TerminationSet: tutte le condizioni di vittoria e sconfitta. "
    "5. Descrizione del LevelMapping: cosa rappresenta ogni simbolo del livello. "
    "6. Nessuna deduzione non presente nel codice. Nessun esempio, nessun commento, "
    "nessuna interpretazione del gameplay. "
    "7. Linguaggio tecnico e preciso, stile documentazione. "
    "Restituisci solo la descrizione testuale, senza citare il codice o ripeterlo. "
    "Ora ti fornirò il codice VGDL."
)

# =======================
# CLIENT OPENAI
# =======================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =======================
# CLIENT MONGODB
# =======================

mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
mongo_db_name = os.getenv("MONGODB_DB", "vgdl_db")
mongo_collection_name = os.getenv("MONGODB_COLLECTION", "vgdl_descriptions")

mongo_client = MongoClient(mongo_uri)
db = mongo_client[mongo_db_name]
collection = db[mongo_collection_name]

# =======================
# FUNZIONE: CHIAMATA MODELLO
# =======================

def generate_description_from_vgdl(vgdl_code: str) -> str:
    """
    Invia al modello OpenAI il codice VGDL e restituisce la descrizione testuale.
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": vgdl_code
            },
        ],
        temperature=0.0,  # per massima aderenza e ripetibilità
    )

    return response.choices[0].message.content.strip()

# =======================
# FUNZIONE: PROCESSA UN FILE
# =======================

def process_file(filepath: str):
    """
    Legge il file VGDL, genera la descrizione con il modello e
    salva input+output su MongoDB.
    """
    filename = os.path.basename(filepath)

    # Leggi il contenuto del file
    with open(filepath, "r", encoding="utf-8") as f:
        vgdl_code = f.read()

    # Genera descrizione dal modello
    print(f"[*] Elaboro file: {filename}")
    description = generate_description_from_vgdl(vgdl_code)

    # Crea il documento da salvare
    document = {
        "filename": filename,
        "filepath": filepath,
        "vgdl_code": vgdl_code,
        "description": description,
        "created_at": datetime.datetime.utcnow(),
    }

    # Inserisci su MongoDB
    result = collection.insert_one(document)
    print(f"[+] Salvato su MongoDB con _id = {result.inserted_id}")

# =======================
# MAIN LOOP
# =======================

def main():
    # Costruisco il pattern completo tipo "vgdl_files/*.txt"
    search_pattern = os.path.join(VGDL_FOLDER, FILE_PATTERN)
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        print(f"Nessun file trovato con pattern: {search_pattern}")
        return

    print(f"Trovati {len(files)} file. Inizio elaborazione...\n")

    for filepath in files:
        try:
            process_file(filepath)
        except Exception as e:
            print(f"[!] Errore con file {filepath}: {e}")

    print("\nElaborazione completata.")

if __name__ == "__main__":
    main()
