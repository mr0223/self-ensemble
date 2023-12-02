import argparse
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, \
    TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset, load_from_disk
import numpy as np
from sklearn.metrics import f1_score

tokenizer = AutoTokenizer.from_pretrained("roberta-base", return_tensors="pt")
tokenizer.add_tokens(["<INPUT>", "<QUERY>"], special_tokens=True)


def tokenized_and_align(examples):
    tokenized_ref = tokenizer(examples["ref"])
    tokenized_sys = tokenizer(examples["baseline_io"])

    labels = []
    is_query = 0
    for sys_id in tokenized_sys["input_ids"]:
        if sys_id == tokenizer.convert_tokens_to_ids("<QUERY>"):
            is_query = 1
            labels.append(-100)
            continue
        if is_query == 0:
            labels.append(-100)
        elif sys_id in tokenizer.all_special_ids:
            labels.append(-100)
        elif sys_id in tokenized_ref["input_ids"]:
            labels.append(0)
        else:
            labels.append(1)

    tokenized_sys["labels"] = labels

    if len(tokenized_sys["input_ids"]) > 512:
        tokenized_sys["input_ids"] = [0] + tokenized_sys["input_ids"][-511:]
        tokenized_sys["attention_mask"] = [1] + tokenized_sys["attention_mask"][-511:]
        tokenized_sys["labels"] = [-100] + tokenized_sys["labels"][-511:]

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
    parser.add_argument('--use_cache', action='store_true')

    args = parser.parse_args()
    print(args)

    multi_num = args.num_beams
    use_cache = args.use_cache

    train_path = f"predict_input/b{multi_num}.train.csv"
    valid_path = f"predict_input/b{multi_num}.valid.csv"
    checkpoint = "roberta-base"
    model_output = f"predict_model/baseline_b{multi_num}"
    train_cache_path = f"predict_input/cache/b{multi_num}.train.tokenized_datasets"
    valid_cache_path = f"predict_input/cache/b{multi_num}.valid.tokenized_datasets"

    if use_cache:
        train_tokenized_datasets = load_from_disk(train_cache_path)
        valid_tokenized_datasets = load_from_disk(valid_cache_path)
    else:
        train_raw_datasets = load_dataset("csv", data_files=train_path, lineterminator='\n')
        valid_raw_datasets = load_dataset("csv", data_files=valid_path, lineterminator='\n')

        train_tokenized_datasets = train_raw_datasets.map(
            tokenized_and_align,
            remove_columns = train_raw_datasets["train"].column_names,
        )
        valid_tokenized_datasets = valid_raw_datasets.map(
            tokenized_and_align,
            remove_columns = valid_raw_datasets["train"].column_names,
        )

        train_tokenized_datasets.save_to_disk(train_cache_path)
        valid_tokenized_datasets.save_to_disk(valid_cache_path)
    
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
    