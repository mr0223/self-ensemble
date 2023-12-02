from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd
import torch
import argparse

def do_beam_search(checkpoint, input_path, output_path, device=torch.device("cuda"), num_beams=4, num_returns=4):
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_tensors="pt")

    with open(input_path) as f:
        input_lines = f.read().split("\n")[:-1]

    input_ids = [tokenizer(input_line, return_tensors="pt").input_ids.to(device) for input_line in input_lines]
    model = model.to(device)

    output_id_lists = [model.generate(
        input_id,
        num_beams=num_beams,
        num_returns=num_returns,
        no_repeat_ngram_size=3,
        remove_invalid_values=True,
    ) for input_id in tqdm(input_ids)]

    output_lines = []
    for output_ids in output_id_lists:
        output_line=[]
        for i in range(num_returns):
            output_line.append(tokenizer.decode(output_ids[i].tolist(), skip_special_tokens=True))
        output_lines.append(output_line)

    output_df = pd.DataFrame(output_lines)

    output_df.to_csv(output_path)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, help="pretrained language model to use")

    parser.add_argument("--input_path", type=str, help="source file")
    parser.add_argument("--output_path", type=str, help="output file")

    parser.add_argument('--num_beams', type=int, default=4,
                        help="Beam size for searching")
    parser.add_argument('--num_returns', type=int, default=4,
                        help="Return sequences size")

    args = parser.parse_args()
    print(args)

    do_beam_search(
        checkpoint = args.checkpoint,
        input_path = args.input_path,
        output_path = args.output_path,
        device = torch.device("cuda"),
        num_beams = args.num_beams,
        num_returns = args.num_returns,
    )

if __name__ == "__main__":
    main()
