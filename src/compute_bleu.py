import json
import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
# import nltk
# nltk.download('wordnet')
from rouge_score import rouge_scorer
from collections import defaultdict
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider

def load_descriptions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_all_scores(gt_dict, pred_dict):
    smoothie = SmoothingFunction().method4
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    bleu1s, bleu2s, bleu3s, bleu4s = [], [], [], []
    meteors, rouges = [], []

    cider_refs = defaultdict(list)
    cider_hyps = {}

    for key in tqdm(gt_dict):
        if key in pred_dict:
            ref = gt_dict[key]
            hyp = pred_dict[key]

            ref_tokens = ref.split()
            hyp_tokens = hyp.split()

            bleu1s.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
            bleu2s.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
            bleu3s.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
            bleu4s.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

            meteors.append(single_meteor_score(ref_tokens, hyp_tokens))  # ← Fixed line
            rouges.append(scorer.score(ref, hyp)['rougeL'].fmeasure)


            cider_refs[key] = [ref]
            cider_hyps[key] = [hyp]
        else:
            print(f"[Warning] Missing key in predictions: {key}")

    cider_score, _ = Cider().compute_score(cider_refs, cider_hyps)

    print("\n--- Evaluation Results ---")
    print(f"BLEU-1: {sum(bleu1s)/len(bleu1s):.4f}")
    print(f"BLEU-2: {sum(bleu2s)/len(bleu2s):.4f}")
    print(f"BLEU-3: {sum(bleu3s)/len(bleu3s):.4f}")
    print(f"BLEU-4: {sum(bleu4s)/len(bleu4s):.4f}")
    print(f"BLEU’ (sum of BLEU-1 to BLEU-4): {(sum(bleu1s)+sum(bleu2s)+sum(bleu3s)+sum(bleu4s))/(len(bleu1s)):.4f}")
    print(f"CIDEr: {cider_score:.4f}")
    print(f"METEOR: {sum(meteors)/len(meteors):.4f}")
    print(f"ROUGE-L: {sum(rouges)/len(rouges):.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compute_bleu.py <ground_truth.json> <predictions.json>")
        sys.exit(1)

    gt_path = sys.argv[1]
    pred_path = sys.argv[2]

    gt_data = load_descriptions(gt_path)
    pred_data = load_descriptions(pred_path)

    compute_all_scores(gt_data, pred_data)
