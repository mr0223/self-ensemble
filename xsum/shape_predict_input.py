import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

def save_csv_multi_xsum_split(
        baseline_output_path, data_type, predict_input_path, num_returns, tokenizer, datasets
    ):
    df = pd.DataFrame(columns=["document_id", "return_id", "ref", "baseline_io"])
    
    input_lines, ref_lines = [], []
    for json_l in datasets[data_type]:
        input_lines.append(json_l["document"])
        ref_lines.append(json_l["summary"])

    output_df = pd.read_csv(baseline_output_path)

    baseline_io_lists, ref_lists, document_id_lists, return_id_lists = [], [], [], []
    for i in range(num_returns):
        output_lines = output_df[f"{i}"].to_list()
        for j in range(len(ref_lines)):
            input_ids = tokenizer.encode(input_lines[j])
            output_ids = tokenizer.encode(output_lines[j])
            input_len = len(input_ids)-2
            output_len = len(output_ids)-2
            if input_len+output_len <= 508:
                baseline_io_lists.append("<INPUT>"+input_lines[j]+"<QUERY>"+str(output_lines[j]))
                ref_lists.append(ref_lines[j])
                document_id_lists.append(j)
                return_id_lists.append(i)
            else:
                baseline_io_lists.append(
                    "<INPUT>"+tokenizer.decode(input_ids[:509-output_len], skip_special_tokens=True)+"<QUERY>"+str(output_lines[j])
                )
                baseline_io_lists.append(
                    "<INPUT>"+tokenizer.decode(input_ids[-509+output_len:], skip_special_tokens=True)+"<QUERY>"+str(output_lines[j])
                )
                ref_lists.append(ref_lines[j])
                ref_lists.append(ref_lines[j])
                document_id_lists.append(j)
                document_id_lists.append(j)
                return_id_lists.append(i)
                return_id_lists.append(i)
    
    df["document_id"] = document_id_lists
    df["return_id"] = return_id_lists
    df["ref"] = ref_lists
    df["baseline_io"] = baseline_io_lists
    
    df.to_csv(predict_input_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--data_type", type=str)
    args = parser.parse_args()

    beam = args.num_beams
    data_type = args.data_type

    xsum_datasets = load_dataset("xsum")
    xsum_datasets["valid"] = xsum_datasets.pop("validation")
    xsum_tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    save_csv_multi_xsum_split(
        baseline_output_path = f"output/baseline/b{beam}.{data_type}.csv", 
        data_type = data_type,
        predict_input_path = f"predict_input/b{beam}.{data_type}.csv", 
        num_returns = beam, 
        tokenizer = xsum_tokenizer,
        datasets = xsum_datasets,
    )


if __name__ == "__main__":
    main()
    