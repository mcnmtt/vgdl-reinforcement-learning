from sacrebleu import sentence_bleu
from bert_score import score

# =========================
# Load files
# =========================

with open(r"gold\lander_gold_description.txt", "r", encoding="utf-8") as f:
    gold = f.read().strip()

with open(r"deepseek\lander_description_deepseek.txt", "r", encoding="utf-8") as f:
    deepseek = f.read().strip()

with open(r"mistral\lander_description_mistral.txt", "r", encoding="utf-8") as f:
    mistral = f.read().strip()

with open(r"phi\lander_description_phi.txt", "r", encoding="utf-8") as f:
    phi = f.read().strip()

# =========================
# BLEU Score
# =========================

bleu_deepseek = sentence_bleu(deepseek, [gold])
bleu_mistral = sentence_bleu(mistral, [gold])
bleu_phi = sentence_bleu(phi, [gold])

# =========================
# BERTScore
# =========================

candidates = [deepseek, mistral, phi]
references = [gold, gold, gold]

P, R, F1 = score(candidates, references, lang="en", verbose=True)

# =========================
# Print Results
# =========================

print("\n===== RESULTS =====\n")

print("BLEU scores")
print(f"DeepSeek BLEU: {bleu_deepseek.score:.4f}")
print(f"Mistral BLEU : {bleu_mistral.score:.4f}")
print(f"Phi BLEU     : {bleu_phi.score:.4f}")

print("\nBERTScore F1")
print(f"DeepSeek BERTScore: {F1[0].item():.4f}")
print(f"Mistral BERTScore : {F1[1].item():.4f}")
print(f"Phi BERTScore     : {F1[2].item():.4f}")