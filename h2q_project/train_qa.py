import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import argparse
from h2q_project.trainer import Trainer as H2QTrainer

def main():
    parser = argparse.ArgumentParser(description="Train a Question Answering model.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name from Hugging Face Transformers")
    parser.add_argument("--dataset_name", type=str, default="squad", help="Dataset name from Hugging Face Datasets")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="output_qa", help="Output directory for saving the model")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every n steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    # Load dataset
    dataset = load_dataset(args.dataset_name)

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx

            idx = len(sequence_ids) - 1
            while sequence_ids[idx] != 1:
                idx -= 1
            context_end = idx

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs


    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Define training arguments
    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     evaluation_strategy="epoch",
    #     learning_rate=args.learning_rate,
    #     per_device_train_batch_size=args.train_batch_size,
    #     per_device_eval_batch_size=args.eval_batch_size,
    #     num_train_epochs=args.num_train_epochs,
    #     weight_decay=0.01,
    #     logging_dir="./logs",
    #     logging_steps=args.logging_steps,
    #     save_strategy="epoch",
    #     load_best_model_at_end=True,
    #     metric_for_best_model="eval_loss",
    #     greater_is_better=False,
    # )

    # Create Trainer instance
    train_dataset = tokenized_datasets["train"].remove_columns([ 'offset_mapping'])
    eval_dataset = tokenized_datasets["validation"].remove_columns(['offset_mapping'])

    trainer = H2QTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=args
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
