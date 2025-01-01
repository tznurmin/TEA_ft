import json
import os
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, TrainerCallback, set_seed
import numpy as np
import datetime
import glob
from datasets import Dataset
from evaluate import load


class LogCallback(TrainerCallback):
    """
    A custom callback to make logging easier
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(logs)
            logs['timestamp'] = datetime.datetime.now().isoformat()
            logs['training_dataset'] = training_path
            logs['test_dataset'] = all_test_path
            logs['r_seed'] = seed_num
            if 'eval_loss' in logs:
                logs['log_entry_type'] = f"test ({current_test_dataset})"
            elif 'loss' in logs:
                logs['log_entry_type'] = 'training'
            elif 'train_runtime' in logs:
                logs['log_entry_type'] = 'training_stats'
            else:
                logs['log_entry_type'] = 'unknown'
            with open(f"logs/{log_filename}", "a") as fp:
                fp.write(f"{json.dumps(logs)}\n")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[l] for l in label if l != -100]  # filter out special tokens
        for label in labels
    ]
    pred_labels = [
        [id2label[pred] for (pred, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=pred_labels, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }



def parse_ner_file(file_path):
    examples = {"tokens": [], "labels": []}
    tokens, labels = [], []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line.split()) < 2:
                if tokens:  # End of an example
                    examples["tokens"].append(tokens)
                    examples["labels"].append(labels)
                    tokens, labels = [], []
            else:
                word, label = line.split()
                tokens.append(word)
                labels.append(label)

    if tokens:
        examples["tokens"].append(tokens)
        examples["labels"].append(labels)

    return examples

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    
    all_labels = examples["labels"]
    new_labels = []

    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        
        for word_idx in word_ids:
            # Remove special tokens etc.
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[labels[word_idx]])
                
        new_labels.append(label_ids)
    
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune with TEA. Select --type (pathogens or strains) and --experiment (augmentation, strategy, mix1, mix2 or mix3). Clone TEA datasets to the project root from: https://github.com/tznurmin/TEA_datasets.git")
    parser.add_argument(
        "--type", 
        type=str, 
        required=True, 
        help="Choose either pathogens or strains"
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        required=True, 
        help="Choose augmentation, strategy, mix1, mix2 or mix3"
    )

    parser.add_argument(
        "--epochs", 
        type=int, 
        required=False, 
        help="Choose the number of epochs to fine-tune (defaults to 2)",
        default=2
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        required=False, 
        help="Define random seed (optional)",
        default=None
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        required=False, 
        help="Choose the used batch size (defaults to 15)",
        default=15
    )

    args = parser.parse_args()
    exp = args.experiment
    exp_type = args.type
    num_epochs = args.epochs
    batch_size =  args.batch_size
    r_seed = args.seed

    if not r_seed is None:
        seed_num = f"{r_seed}"
        set_seed(r_seed)
    else:
        seed_num = 'noseed'

    if 'mix' in exp:
        experiment_dir = f"TEA_datasets/datasets/{exp_type}/mix/training/{exp[-1]}"
    else:
        experiment_dir = f"TEA_datasets/datasets/{exp_type}/{exp}/training"

    test_dir = f"{experiment_dir.split('/training')[0]}/test"

    training_paths = glob.glob(f"{experiment_dir}/*.set")
    for training_path in training_paths:

        start_time = int(datetime.datetime.today().timestamp())

        if training_path.split('_')[-1] == 'training.set':
            number = training_path.split('/')[-2]
        else:
            number = training_path.split('_')[-1]
            number = number.split('.')[0]
        
        all_test_path = f"{test_dir}/all_1v_test_{number}.set"
        none_test_path = f"{test_dir}/unaugmented_1v_test_{number}.set"

        if not os.path.exists(all_test_path):
            print(f"Error: {all_test_path} not found")
            exit()
        
        if not os.path.exists(none_test_path):
            print(f"Error: {none_test_path} not found")
            exit()

        if not os.path.exists(training_path):
            print(f"Error: {training_path} not found")
            exit()
        
        os.makedirs('logs', exist_ok=True)
        log_filename = f"{training_path.replace('/', '_')}_{start_time}.json"

        if os.path.exists(f"logs/{log_filename}"):
            os.remove(f"logs/{log_filename}")

        model_path = "dmis-lab/biobert-base-cased-v1.2"
        
        # Very important step: biobert-base-cased-v1.2 tokenizer is currently NOT cased by default
        # copy the configuration file from biobert-base-cased-v1.1 repository for maximum compatibility
        tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False, model_max_length=512)


        seqeval = load("seqeval")


        train_examples = parse_ner_file(training_path)
        all_test_examples = parse_ner_file(all_test_path)
        none_test_examples = parse_ner_file(none_test_path)

        train_dataset = Dataset.from_dict(train_examples)
        all_test_dataset = Dataset.from_dict(all_test_examples)
        none_test_dataset = Dataset.from_dict(none_test_examples)

        unique_labels = set()
        for label_seq in train_dataset["labels"]:
            for label in label_seq:
                unique_labels.add(label)

        unique_labels = sorted(list(unique_labels))  # sort to have consistent ordering
        label2id = {label: i for i, label in enumerate(unique_labels)}
        id2label = {i: label for label, i in label2id.items()}


        
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(unique_labels),
            id2label=id2label,
            label2id=label2id,
        )

        train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
        train_dataset.remove_columns(["tokens"])

        all_test_dataset = all_test_dataset.map(tokenize_and_align_labels, batched=True)
        all_test_dataset.remove_columns(["tokens"])

        none_test_dataset = none_test_dataset.map(tokenize_and_align_labels, batched=True)
        none_test_dataset.remove_columns(["tokens"])



        training_args = TrainingArguments(
            num_train_epochs=num_epochs,
            output_dir="./saved_models",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_strategy="steps",
            logging_steps=30,
            logging_dir='logs',
            save_strategy="no",
            learning_rate=5e-5,
        )

        if not r_seed is None:
            training_args.seed = r_seed

        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[LogCallback()],
        )

        trainer.train()
        current_test_dataset = 'all'
        trainer.evaluate(all_test_dataset)
        current_test_dataset = 'none'
        trainer.evaluate(none_test_dataset)
    