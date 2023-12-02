from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
import numpy as np
import math
import torch
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", type=str, help="output csv file")
    parser.add_argument('--num_beams', type=int, default=10)
    parser.add_argument('--num_returns', type=int, default=10)
    parser.add_argument('--data_type', type=str, help="train or valid or test")
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    print(args)

    checkpoint = "facebook/bart-large-cnn"
    device=torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model = model.to(device)

    datasets = load_dataset("cnn_dailymail", "3.0.0")
    datasets["valid"] = datasets.pop("validation")

    input_lines, ref_lines = [], []
    for json_l in datasets[args.data_type]:
        input_lines.append(json_l["article"])
        ref_lines.append(json_l["highlights"])

    output_texts = []
    num_iters = math.ceil(len(input_lines)/args.batch_size)
    for i in tqdm(range(num_iters)):
        batch_input_lines = input_lines[i*args.batch_size:(i+1)*args.batch_size]
        batch_input_ids = tokenizer(batch_input_lines, return_tensors="pt", truncation=True, padding=True).to(device)
        batch_output_ids = model.generate(
            **batch_input_ids,
            num_beams=args.num_beams,
            num_return_sequences=args.num_returns,
            no_repeat_ngram_size=3,
            remove_invalid_values=True,
            max_length=1024,
        )
        batch_output_texts = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)
        output_texts += batch_output_texts

    output_texts_reshaped = np.array(output_texts).reshape(-1,args.num_returns).tolist()
    output_df = pd.DataFrame(output_texts_reshaped)
    output_df.to_csv(args.output_path)


if __name__ == "__main__":
    main()
