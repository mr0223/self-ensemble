from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
from collections import Counter
import pandas as pd
import torch
import argparse
import pickle


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


def make_one_constraint_withinput(
    line, model, bert_tokenizer, device
    ):
    model = model.to(device)

    input_tokens = bert_tokenizer(line)
    if len(input_tokens["input_ids"]) > 512:
        line = bert_tokenizer.decode(input_tokens["input_ids"][-511:-1])
    
    line_ids = bert_tokenizer(line, return_tensors="pt", truncation=True).input_ids.to(device)
    classification_results = torch.argmax(model(line_ids).logits[0], dim=1)

    for (i, line_id) in enumerate(line_ids[0]):
        if line_id == bert_tokenizer.convert_tokens_to_ids("<QUERY>"):
            present_token_id = i+1
            break

    good_ids_bart, bad_ids_bart = [], []
    while present_token_id < len(line_ids[0]):
        if line_ids[0][present_token_id] in bert_tokenizer.all_special_ids:
            present_token_id += 1
        elif classification_results[present_token_id]==0:
            good_ids_bart.append(line_ids[0][present_token_id].cpu().tolist())
            present_token_id += 1
        elif classification_results[present_token_id]==1:
            bad_ids_bart.append(line_ids[0][present_token_id].cpu().tolist())
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


def make_one_constraint_withinput_list(
    one_line_list, model, bert_tokenizer, device
    ):
    one_constraint_raws = []
    for one_line in one_line_list:
        one_constraint_raw, _ = make_one_constraint_withinput(one_line, model, bert_tokenizer, device)
        one_constraint_raws.append(one_constraint_raw)
    
    p_ids_multi_set, n_ids_multi_set = [], []
    for one_constraint_raw in one_constraint_raws:
        p_ids, n_ids = [], []
        for one_constraint in one_constraint_raw:
            if one_constraint[0][1]:
                p_ids.append(one_constraint[0][0][0])
            elif not one_constraint[0][1]:
                n_ids.append(one_constraint[0][0][0])
        p_ids_multi_set.append(set(p_ids))
        n_ids_multi_set.append(set(n_ids))
    
    p_ids_union = p_ids_multi_set[0]
    n_ids_union = n_ids_multi_set[0]
    for p_ids_set in p_ids_multi_set:
        p_ids_union = p_ids_union.union(p_ids_set)
    for n_ids_set in n_ids_multi_set:
        n_ids_union = n_ids_union.union(n_ids_set)

    pn_ids = n_ids_union.intersection(p_ids_union)
    p_ids_integrated_union = list(p_ids_union - pn_ids)
    n_ids_integrated_union = list(n_ids_union - pn_ids)

    constraint_raw_union, constraint_len_union = convert_ids_to_raws(p_ids_integrated_union, n_ids_integrated_union)

    return constraint_raw_union, constraint_len_union


def make_one_constraint_withinput_multi(
    one_line_summarized, model, bert_tokenizer, device
    ):
    one_constraint_raws = []
    for one_line_list in one_line_summarized:
        one_constraint_raw, _ = make_one_constraint_withinput_list(one_line_list, model, bert_tokenizer, device)
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
    
    for integrated_ids in [p_ids_integrated_majority, n_ids_integrated_majority]:
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


def make_whole_constraint_withinput_multi(
    io_df, model, bert_tokenizer, device
    ):
    document_num = max(io_df["document_id"].to_list()) + 1
    return_num = max(io_df["return_id"].to_list()) + 1
    whole_line_summarized = [
        [io_df[(io_df["document_id"]==i) & (io_df["return_id"]==j)]["baseline_io"].to_list() for j in range(return_num)] \
        for i in range(document_num)
    ]

    constraint_raw_nums_maj_uni = [
        make_one_constraint_withinput_multi(one_line_summarized, model, bert_tokenizer, device) \
        for one_line_summarized in tqdm(whole_line_summarized)
    ]

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
    constraint_path_majority = f"constraint/both/b{return_num}.majority.{data_type}.pkl"
    constraint_path_majority_p = f"constraint/positive/b{return_num}.majority.{data_type}.pkl"
    constraint_path_majority_n = f"constraint/negative/b{return_num}.majority.{data_type}.pkl"
    # constraint_path_union = f"constraint/both/b{return_num}.union.{data_type}.pkl"
    device=torch.device("cuda")

    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    bert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    io_df = pd.read_csv(predict_input_path)
    
    constraint_raws_majority, constraint_nums_majority, constraint_raws_union, constraint_nums_union \
            = make_whole_constraint_withinput_multi(io_df, model, bert_tokenizer, device)

    with open(constraint_path_majority, "wb") as f:
        pickle.dump(constraint_raws_majority, f)
    # with open(constraint_path_union, "wb") as f:
    #     pickle.dump(constraint_raws_union, f)

    make_pn_constraints(constraint_path_majority, constraint_path_majority_p, constraint_path_majority_n)


if __name__ == "__main__":
    main()
