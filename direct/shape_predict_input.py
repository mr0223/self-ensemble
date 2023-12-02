import pandas as pd
import argparse

def save_csv_multi(input_path, baseline_output_path, ref_path, predict_input_path, num_returns):
    df = pd.DataFrame(columns=["ref", "baseline_io"])
    with open(ref_path) as f:
        ref_lines = f.read().split("\n")[:-1]
    with open(input_path) as f:
        input_lines = f.read().split("\n")[:-1]
    output_df = pd.read_csv(baseline_output_path)

    output_lists, input_lists, ref_lists = [], [], []
    for i in range(num_returns):
        output_lists += output_df[f"{i}"].to_list()
        input_lists += input_lines
        ref_lists += ref_lines
    
    df["ref"] = ref_lists
    df["baseline_io"] = ["<INPUT>"+input_list+"<QUERY>"+str(output_list) for (input_list, output_list) in zip(input_lists, output_lists)]
    
    df.to_csv(predict_input_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, help="source file")
    parser.add_argument("--baseline_output_path", type=str, help="beam search output file")
    parser.add_argument("--ref_path", type=str, help="reference file")
    parser.add_argument("--predict_input_path", type=str)
    parser.add_argument("--num_beams", type=int, default=10)
    args = parser.parse_args()

    save_csv_multi(
        input_path = args.input_path, 
        baseline_output_path = args.baseline_output_path,
        ref_path = args.ref_path, 
        predict_input_path = args.predict_input_path,
        num_returns = args.num_beams, 
    )


if __name__ == "__main__":
    main()
    