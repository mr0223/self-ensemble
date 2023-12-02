import argparse
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, \
    TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score

tokenizer = AutoTokenizer.from_pretrained("roberta-base", return_tensors="pt")
tokenizer.add_tokens(["<INPUT>", "<QUERY>", "<user>", "<system>", "<query>"], special_tokens=True)


def truncate_baseline(examples):
    inputs = examples["baseline_io"]
    input_tokens = tokenizer(inputs)

    truncated_inputs = {}

    if len(input_tokens["input_ids"]) > 512:
        truncated_inputs["baseline_io"] = tokenizer.decode(input_tokens["input_ids"][-511:-1])
    else:
        truncated_inputs["baseline_io"] = inputs

    return truncated_inputs


def tokenized_and_align(examples):
    tokenized_ref = tokenizer(examples["ref"])
    tokenized_sys = tokenizer(examples["baseline_io"])

    labels = []
    for i in range(len(examples["ref"])):
        one_labels = []
        is_query = 0
        for sys_id in tokenized_sys["input_ids"][i]:
            if sys_id == tokenizer.convert_tokens_to_ids("<QUERY>"):
                is_query = 1
                one_labels.append(-100)
                continue
            if is_query == 0:
                one_labels.append(-100)
            elif sys_id in tokenizer.all_special_ids:
                one_labels.append(-100)
            elif sys_id in tokenized_ref["input_ids"][i]:
                one_labels.append(0)
            else:
                one_labels.append(1)
        labels.append(one_labels)

    tokenized_sys["labels"] = labels
    return tokenized_sys


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    predictions_flatten = predictions.flatten()
    labels_flatten = np.array(labels).flatten()

    preds_binary, refs_binary = [], []
    true0, true1, false0, false1 = 0, 0, 0, 0
    for (prediction, label) in zip(predictions_flatten, labels_flatten):
        if (prediction==0) and (label==0):
            true0+=1
            preds_binary.append(prediction)
            refs_binary.append(label)
        elif (prediction==0) and (label==1):
            false0+=1
            preds_binary.append(prediction)
            refs_binary.append(label)
        elif (prediction==1) and (label==1):
            true1+=1
            preds_binary.append(prediction)
            refs_binary.append(label)
        elif (prediction==1) and (label==0):
            false1+=1
            preds_binary.append(prediction)
            refs_binary.append(label)

    return {
        "true-0": true0, 
        "false-0": false0, 
        "true-1": true1, 
        "false-1": false1,
        "f1-macro": f1_score(preds_binary, refs_binary, average="macro"),
        "f1-micro": f1_score(preds_binary, refs_binary, average="micro"),
        "f1-weighted": f1_score(preds_binary, refs_binary, average="weighted")
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_beams', type=int, default=10)
    parser.add_argument('--task', type=str, help="D2I or I2D")

    args = parser.parse_args()
    print(args)

    multi_num = args.num_beams
    task = args.task

    if task == "D2I":
        output_type = "w.indirect"
    elif task == "I2D":
        output_type = "w.direct"

    train_path = f"predict_input/baseline/b{multi_num}.train.{output_type}.csv"
    valid_path = f"predict_input/baseline/b{multi_num}.valid.{output_type}.csv"
    checkpoint = "roberta-base"
    model_output = f"predict_model/{task}_w_baseline_b{multi_num}"

    train_raw_datasets = load_dataset("csv", data_files=train_path)
    valid_raw_datasets = load_dataset("csv", data_files=valid_path)

    train_truncated_datasets = train_raw_datasets.map(
        truncate_baseline,
    )
    valid_truncated_datasets = valid_raw_datasets.map(
        truncate_baseline,
    )

    train_tokenized_datasets = train_truncated_datasets.map(
        tokenized_and_align,
        batched = True,
        remove_columns = train_raw_datasets["train"].column_names,
    )
    valid_tokenized_datasets = valid_truncated_datasets.map(
        tokenized_and_align,
        batched = True,
        remove_columns = valid_raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=3, ignore_mismatched_sizes=True)

    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=model_output,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        metric_for_best_model="f1-macro",
        load_best_model_at_end=True,
        greater_is_better=True,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=20,
        logging_dir=model_output+"/logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_datasets["train"],
        eval_dataset=valid_tokenized_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()


if __name__ == "__main__":
    main()
