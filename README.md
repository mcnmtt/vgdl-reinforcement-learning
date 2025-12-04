# DATASET
Per la generazione del dataset:
1) Convertire i **`.py`** in **`.txt`** contenenti solo VGDL (BasicGame) eseguendo **`dataset/extract_vgdl.py`**
2) Generare descrizioni VGDL tramite GPT-5 con OpenAI API e salvare in DB NoSQL
    - Configurare **`.env`**
    - Avviare **`dataset/create_db.py`**