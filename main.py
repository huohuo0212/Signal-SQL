"""
main.py - Universal Entrypoint for Signal-SQL / Text-to-SQL Prompt Engineering
=============================================================================
Usage:
  # 1. Preview generated Few-Shot Prompt:
  python main.py --run_mode prompt_only --selector_type SIGNAL_DETECTION --k_shot 7

  # 2. Run evaluation on Spider with GPT-4:
  python main.py --run_mode inference --dataset spider --selector_type SIGNAL_DETECTION --k_shot 7 --model gpt-4
"""

import os
import sys
import argparse
import json
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_builder import load_data
from utils.enums import REPR_TYPE, EXAMPLE_TYPE, SELECTOR_TYPE, LLM
from prompt.prompt_builder import prompt_factory
from llm.chatgpt import init_chatgpt, ask_llm


def parse_args():
    parser = argparse.ArgumentParser(description="Signal-SQL Evaluation & Prompt Pipeline")
    parser.add_argument("--dataset", type=str, default="spider", choices=["spider", "bird", "realistic"],
                        help="Benchmark dataset to run on")
    parser.add_argument("--data_path", type=str, default="./dataset",
                        help="Root path containing dataset files")
    parser.add_argument("--repr_type", type=str, default=REPR_TYPE.CODE_REPRESENTATION,
                        help="Database representation style (e.g. SQL, TEXT, NUMBERSIGN, SQLCOT)")
    parser.add_argument("--selector_type", type=str, default=SELECTOR_TYPE.SIGNAL_DETECTION,
                        help="Example selector (e.g. SIGNAL_DETECTION, EUCDISMASKPRESKLSIMTHR, BM25)")
    parser.add_argument("--example_format", type=str, default=EXAMPLE_TYPE.QA,
                        help="Example formatting style (e.g. QA, ONLYSQL, COMPLETE)")
    parser.add_argument("--k_shot", type=int, default=7,
                        help="Number of in-context few-shot examples (k-shot)")
    parser.add_argument("--model", type=str, default=LLM.GPT_4,
                        help="Target LLM for SQL generation (e.g. gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save generated SQL results")
    parser.add_argument("--mini_set", action="store_true",
                        help="Evaluate on mini development set for fast verification")
    parser.add_argument("--cross_domain", action="store_true", default=True,
                        help="Enforce cross-domain example selection (exclude same database)")
    parser.add_argument("--run_mode", type=str, default="prompt_only", choices=["prompt_only", "inference"],
                        help="Execution mode: 'prompt_only' inspects prompts; 'inference' calls LLM")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Max number of test questions to process (for debugging)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print(f"  Signal-SQL Execution Pipeline: [{args.dataset.upper()}]")
    print(f"  Mode:            {args.run_mode}")
    print(f"  Selector:        {args.selector_type}")
    print(f"  Representation:  {args.repr_type}")
    print(f"  Format:          {args.example_format} | {args.k_shot}-Shot")
    print(f"  Model:           {args.model}")
    print("=" * 65)

    # 1. Check dataset availability
    dataset_dir = os.path.join(args.data_path, args.dataset)
    if not os.path.exists(dataset_dir):
        print(f"\n[!] Dataset directory not found at: {dataset_dir}")
        print("    To run full benchmarks, please download the Spider/BIRD dataset into './dataset/'.")
        print("    For a quick self-contained test, run: python run_demo.py")
        sys.exit(0)

    # 2. Load dataset
    print("\n[1/3] Loading dataset...")
    data = load_data(args.dataset, args.data_path)
    test_data = data.get_test_json(mini_set=args.mini_set)
    print(f"      Loaded {len(test_data)} test items.")

    # 3. Instantiate Prompt Builder Factory
    print("\n[2/3] Initializing Prompt Factory...")
    PromptClass = prompt_factory(
        repr_type=args.repr_type,
        k_shot=args.k_shot,
        example_format=args.example_format,
        selector_type=args.selector_type
    )
    prompt_builder = PromptClass(data=data, tokenizer="gpt-3.5-turbo")

    # 4. Process Samples
    print(f"\n[3/3] Processing samples (max: {args.max_samples})...")
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.dataset}_{args.k_shot}-SHOT_{args.selector_type}.txt")

    if args.run_mode == "inference":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("[Error] OPENAI_API_KEY not found in environment. Please set it in .env or shell.")
            sys.exit(1)
        init_chatgpt(api_key, None, args.model)

    saved_sqls = []
    samples_to_run = test_data[:args.max_samples] if args.max_samples > 0 else test_data

    for idx, target in enumerate(tqdm(samples_to_run, desc="Generating")):
        prompt_info = prompt_builder.format(
            target=target,
            max_seq_len=4096,
            max_ans_len=512,
            scope_factor=1,
            cross_domain=args.cross_domain
        )
        prompt_text = prompt_info["prompt"]

        if args.run_mode == "prompt_only":
            if idx == 0:
                print("\n" + "=" * 30 + " Sample Prompt [Index 0] " + "=" * 30)
                print(prompt_text[:1200] + ("\n... [truncated for display] ..." if len(prompt_text) > 1200 else ""))
                print("=" * 75)
                print(f"-> Estimated Prompt Tokens: {prompt_info.get('prompt_tokens', 'N/A')}")
                print(f"-> Valid Examples Injected: {prompt_info.get('n_examples', 0)}")
            continue

        # Inference mode
        res = ask_llm(args.model, [prompt_text], temperature=0.0, n=1)
        pred_sql = res["response"][0] if isinstance(res["response"], list) else res["response"]
        saved_sqls.append(pred_sql.strip().replace("\n", " "))

    if args.run_mode == "inference" and saved_sqls:
        with open(out_file, "w", encoding="utf-8") as f:
            for s in saved_sqls:
                f.write(s + "\n")
        print(f"\n[OK] Predictions saved to: {out_file}")
    elif args.run_mode == "prompt_only":
        print(f"\n[OK] Prompt generation test passed for {len(samples_to_run)} samples.")


if __name__ == "__main__":
    main()
