import subprocess
import argparse

def run_command(cmd):
    print(f"\n Running command: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f" Command failed: {cmd}")
    else:
        print(f" Finished: {cmd}")

def main():
    parser = argparse.ArgumentParser(description="Run training or evaluation scripts")
    parser.add_argument("--mode", choices=["train", "eval", "generative"], required=True, help="Which module to run")
    parser.add_argument("--model", required=True, help="Model name (e.g., bert, convnextv2, gpt4-text, etc.)")
    parser.add_argument("--extra_args", default="", help="Extra arguments to pass to the script")
    args = parser.parse_args()
    
    script_map = {
    "train": {
        "bert": "train/train_bert.py",
        "convnextv2": "train/train_convnextv2.py",
        "efficientnet": "train/train_efnet.py",
        # "dinov2": "train/train_dinov2.py",
        "modernbert": "train/train_modernbert.py",
        "t5": "train/train_t5.py",
        "vanillamultimodal_mbert_efnet": "train/train_efnetmbert.py",  # vanilla
        # "vanilla_modernbert_convnextv2": "train/train_vanilla_modernbert_convnextv2.py",
        "vanillamultimodal_bert_efnet": "train/train_vanillamultimodal_bert_efnet.py",
        # "llava_next": "train/train_llava_next.py"
    },
    "eval": {
        "bert": "eval/eval_bert.py",
        "convnextv2": "eval/eval_convnextv2.py",
        "efficientnet": "eval/eval_efnet.py",
        # "dinov2": "eval/eval_dinov2.py",
        "modernbert": "eval/eval_modernbert.py",
        "t5": "eval/eval_t5.py",
        "vanillamultimodal_mbert_efnet": "eval/eval_efnetmbert.py",  # vanilla
        # "vanilla_modernbert_convnextv2": "eval/eval_vanilla_modernbert_convnextv2.py",
        "vanillamultimodal_bert_efnet": "eval/eval_vanillamultimodal_bert_efnet.py",
        # "llava_next": "eval/eval_llava_next.py"
    },
    "generative": {
        "gpt4-text": "generative/train_eval_gpt4.py",
        "gpt4-image": "generative/train_eval_gpt4_image.py",
        "gpt4-multimodal": "generative/train_eval_gpt4_multimodal.py",
        "gpt4-text": "generative/train_eval_gpt4_text.py",
        # "gpt4-text-zero-shot": "generative/train_eval_gpt4_text_zero_shot.py",
        "llava-image": "generative/train_eval_llava_image.py",
        "llava-multimodal": "generative/train_eval_llava_multimodal.py"
    }
}


    if args.model not in script_map[args.mode]:
        print(f" Unknown model '{args.model}' for mode '{args.mode}'")
        return

    script_path = script_map[args.mode][args.model]
    cmd = f"python {script_path} {args.extra_args}"
    run_command(cmd)

if __name__ == "__main__":
    main()
