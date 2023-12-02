from transformers import AutoTokenizer
import pandas as pd
from collections import Counter
from tqdm import tqdm
import pickle
import argparse


def convert_ids_to_raws(p_ids, n_ids):
    if (len(p_ids)+len(n_ids))==0:
        constraint_raw = [[([0], True)]]
        constraint_len = 0
    else:
        constraint_raw = []
        constraint_len = len(p_ids)+len(n_ids)
        if len(p_ids)!=0:
            for p_id in p_ids:
                constraint_raw.append([([p_id], True)])
        else:
            constraint_raw.append([([0], True)])
        if len(n_ids)!=0:
            for n_id in n_ids:
                constraint_raw.append([([n_id], False)])
    
    return constraint_raw, constraint_len


def make_one_constraint_correct(
    line, ref_line, model, bert_tokenizer
    ):
    line_ids = bert_tokenizer(line).input_ids
    line_tokens = bert_tokenizer.convert_ids_to_tokens(line_ids)
    ref_line_ids = bert_tokenizer(ref_line).input_ids
    
    classification_results = []
    for line_id in line_ids:
        if line_id in bert_tokenizer.all_special_ids:
            classification_results.append(-100)
        elif line_id in ref_line_ids:
            classification_results.append(0)
        else:
            classification_results.append(1)

    for (i, line_id) in enumerate(line_ids):
        if line_id == bert_tokenizer.convert_tokens_to_ids("<QUERY>"):
            present_token_id = i+1
            break

    good_ids_bart, bad_ids_bart = [], []
    while present_token_id < len(line_ids):
        if line_ids[present_token_id] in bert_tokenizer.all_special_ids:
            present_token_id += 1
        elif classification_results[present_token_id]==0:
            good_ids_bart.append(line_ids[present_token_id])
            present_token_id += 1
        elif classification_results[present_token_id]==1:
            bad_ids_bart.append(line_ids[present_token_id])
            present_token_id += 1
        else:
            present_token_id += 1

    p_constraint_id, n_constraint_id = [], []
    for good_id_bart in good_ids_bart:
        if good_id_bart not in p_constraint_id:
            p_constraint_id.append(good_id_bart)
    for bad_id_bart in bad_ids_bart:
        if bad_id_bart not in n_constraint_id:
            n_constraint_id.append(bad_id_bart)
    
    constraint_raw, constraint_len = convert_ids_to_raws(p_constraint_id, n_constraint_id)

    return constraint_raw, constraint_len


def make_one_constraint_correct_multi(
    one_line_summarized, one_ref_line_summarized, bert_tokenizer
    ):
    one_constraint_raws = []
    for (one_line, one_ref_line) in zip(one_line_summarized, one_ref_line_summarized):
        one_constraint_raw, _ = make_one_constraint_correct(one_line, one_ref_line, bert_tokenizer)
        one_constraint_raws.append(one_constraint_raw)
    
    p_ids_multi, n_ids_multi = [], []
    p_ids_multi_set, n_ids_multi_set = [], []
    for one_constraint_raw in one_constraint_raws:
        p_ids, n_ids = [], []
        for one_constraint in one_constraint_raw:
            if one_constraint[0][1]:
                p_ids.append(one_constraint[0][0][0])
            elif not one_constraint[0][1]:
                n_ids.append(one_constraint[0][0][0])
        p_ids_multi.append(p_ids)
        n_ids_multi.append(n_ids)
        p_ids_multi_set.append(set(p_ids))
        n_ids_multi_set.append(set(n_ids))
    
    p_ids_multi_flatten = sum(p_ids_multi, [])
    n_ids_multi_flatten = sum(n_ids_multi, [])
    
    all_ids = []
    for one_id in (p_ids_multi_flatten + n_ids_multi_flatten):
        if one_id not in all_ids:
            all_ids.append(one_id)
    p_counter = Counter(p_ids_multi_flatten)
    n_counter = Counter(n_ids_multi_flatten)
    p_ids_integrated_majority, n_ids_integrated_majority = [], []
    for one_id in all_ids:
        p_count = p_counter[one_id]
        n_count = n_counter[one_id]
        if p_count > n_count:
            p_ids_integrated_majority.append(one_id)
        elif p_count < n_count:
            n_ids_integrated_majority.append(one_id)
    
    integrated_idss = [p_ids_integrated_majority, n_ids_integrated_majority]
    for integrated_ids in integrated_idss:
        try:
            integrated_ids.remove(0)
        except ValueError:
            pass
    
    p_ids_union = p_ids_multi_set[0]
    n_ids_union = n_ids_multi_set[0]
    for p_ids_set in p_ids_multi_set:
        p_ids_union = p_ids_union.union(p_ids_set)
    for n_ids_set in n_ids_multi_set:
        n_ids_union = n_ids_union.union(n_ids_set)

    pn_ids = n_ids_union.intersection(p_ids_union)
    p_ids_integrated_union = list(p_ids_union - pn_ids - {0})
    n_ids_integrated_union = list(n_ids_union - pn_ids - {0})

    constraint_raw_majority, constraint_len_majority = convert_ids_to_raws(p_ids_integrated_majority, n_ids_integrated_majority)
    constraint_raw_union, constraint_len_union = convert_ids_to_raws(p_ids_integrated_union, n_ids_integrated_union)

    return constraint_raw_majority, constraint_len_majority, constraint_raw_union, constraint_len_union


def make_whole_constraint_correct_multi(
    whole_line, whole_ref_line, return_num, bert_tokenizer
    ):
    original_line_num = int(len(whole_line) / return_num)
    whole_line_summarized = []
    whole_ref_line_summarized = []
    for original_line_id in range(original_line_num):
        whole_line_summarized.append([whole_line[original_line_id + return_id*original_line_num] for return_id in range(return_num)])
        whole_ref_line_summarized.append([whole_ref_line[original_line_id + return_id*original_line_num] for return_id in range(return_num)])

    constraint_raw_nums_maj_uni = []
    for i in tqdm(range(len(whole_line_summarized))):
        constraint_raw_nums_maj_uni.append(
            make_one_constraint_correct_multi(whole_line_summarized[i], whole_ref_line_summarized[i], bert_tokenizer)
        )

    constraint_raws_majority = [raw_num[0] for raw_num in constraint_raw_nums_maj_uni]
    constraint_nums_majority = [raw_num[1] for raw_num in constraint_raw_nums_maj_uni]
    constraint_raws_union = [raw_num[2] for raw_num in constraint_raw_nums_maj_uni]
    constraint_nums_union = [raw_num[3] for raw_num in constraint_raw_nums_maj_uni]

    return constraint_raws_majority, constraint_nums_majority, constraint_raws_union, constraint_nums_union


def make_pn_constraints(input_path, p_out_path, n_out_path):
    with open(input_path, "rb") as f:
        constraints_list = pickle.load(f)

    p_constraints_list, n_constraints_list = [], []
    for constraints in constraints_list:
        p_constraints, n_constraints = [], [[([0], True)]]
        for constraint in constraints:
            if constraint[0][1]:
                p_constraints.append(constraint)
            else:
                n_constraints.append(constraint)
        p_constraints_list.append(p_constraints)
        n_constraints_list.append(n_constraints)
    
    with open(p_out_path, "wb") as f:
        pickle.dump(p_constraints_list, f)
    with open(n_out_path, "wb") as f:
        pickle.dump(n_constraints_list, f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--return_num', type=int, default=10)
    parser.add_argument('--data_type', type=str, help="train or valid or test")
    parser.add_argument('--checkpoint', type=str, help="predict model path")

    args = parser.parse_args()
    print(args)

    return_num = args.return_num
    data_type = args.data_type
    checkpoint = args.checkpoint

    predict_input_path = f"predict_input/b{return_num}.{data_type}.csv"
    constraint_path_majority = f"constraint/_both/b{return_num}.majority.{data_type}.pkl"
    constraint_path_majority_p = f"constraint/_positive/b{return_num}.majority.{data_type}.pkl"
    constraint_path_majority_n = f"constraint/_negative/b{return_num}.majority.{data_type}.pkl"
    # constraint_path_union = f"constraint/_both/b{return_num}.union.{data_type}.pkl"

    bert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model_input_lines = pd.read_csv(predict_input_path)["baseline_io"].to_list()
    ref_lines = pd.read_csv(predict_input_path)["ref"].to_list()

    constraint_raws_majority, constraint_nums_majority, \
        constraint_raws_union, constraint_nums_union \
            = make_whole_constraint_correct_multi(model_input_lines, ref_lines, return_num, bert_tokenizer)

    with open(constraint_path_majority, "wb") as f:
        pickle.dump(constraint_raws_majority, f)
    # with open(constraint_path_union, "wb") as f:
    #     pickle.dump(constraint_raws_union, f)

    make_pn_constraints(constraint_path_majority, constraint_path_majority_p, constraint_path_majority_n)


if __name__ == "__main__":
    main()