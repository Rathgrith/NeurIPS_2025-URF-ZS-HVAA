import json
import openai
import os
from time import sleep
from tqdm import tqdm
import argparse
# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # get API key from environment variable
if not openai.api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# System prompt (unchanged)
SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the generative outputs for video-based pairs. "
    "You will be given two answers, one reference ground truth and one our generated, but this does not mean "
    "that the reference GT is the only answer. Your task is to give the score of the predicted answers."
)

# Build the evaluation prompt for a given dimension
def build_evaluation_prompt(gt, pred, dimension):
    return (
        "### Video Description Generation\n"
        "Please evaluate the following video-based video description pair:\n"
        f"Reference: {gt}\n"
        f"Ours: {pred}\n"
        f"Provide your evaluation only as a {dimension} score where the {dimension} score is a FLOAT value "
        "between 0 and 1, with 1 indicating the highest level of "
        f"{dimension}. Please generate the response in the form of a Python dictionary string with key 'score', "
        "where its value is the {dimension} score in FLOAT, not STRING. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. "
        "Only provide the Python dictionary string. For example, your response should look like this: {'score': 0.675}."
    )

# Query GPT-4 for a single dimension score
def evaluate_dimension(gt, pred, dimension, max_retries=3):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_evaluation_prompt(gt, pred, dimension)},
    ]
    for attempt in range(max_retries):
        try:
            res = openai.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                temperature=0.0
            )
            reply = res.choices[0].message.content.strip()
            score_dict = eval(reply)  # assumes strict format
            return score_dict.get("score")
        except Exception as e:
            print(f"Attempt {attempt+1} for {dimension} failed: {e}")
            sleep(1)
    return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate generated descriptions against ground truth')
    parser.add_argument('--gt_path', type=str, default="./data/ucf_crime/video_summaries.json",
                        help='Path to ground truth annotations')
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Path to predicted descriptions')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save evaluation results')
    args = parser.parse_args()

    # Load GT
    with open(args.gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    # Load predictions
    with open(args.pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)


        
    dimensions = ["Reasonability", "Detail", "Consistency"]
    results = {}
    count = 0
    for vid, pred_desc in tqdm(pred_data.items()):
        count += 1
        if vid not in gt_data:
            print(f"Skipping {vid}: no GT")
            continue
        # if count == 2:
        #     break
        gt_desc = gt_data[vid]
        print(f"Evaluating {vid}...")
        scores = {}
        for d in dimensions:
            sc = evaluate_dimension(gt_desc, pred_desc, d)
            scores[d] = sc if sc is not None else "N/A"
            print(f"  {d}: {scores[d]}")
        results[vid] = scores

    # Compute mean scores
    mean_scores = {}
    for d in dimensions:
        vals = [scores[d] for scores in results.values() if isinstance(scores[d], (int, float))]
        mean_scores[d] = sum(vals) / len(vals) if vals else None

    # Attach and print
    results["mean_scores"] = mean_scores
    print("\nMean scores:")
    for d, m in mean_scores.items():
        print(f"  {d}: {m:.4f}" if m is not None else f"  {d}: N/A")

    # Save all results
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    main()