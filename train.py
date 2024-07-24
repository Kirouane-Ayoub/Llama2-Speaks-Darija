import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import settings
from dataset import TextDataset, get_tokenizer
from model import Transformer
from settings import ModelArgs, TrainArgs


# train_model function
def train_model(model, train_loader, eval_loader, train_args, tokenizer):
    model = model.to(train_args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, step / train_args.warmup_steps)
    )

    for epoch in range(train_args.n_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids, labels = batch
            input_ids, labels = (
                input_ids.to(train_args.device),
                labels.to(train_args.device),
            )

            outputs = model(input_ids)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs.view(-1, tokenizer.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % train_args.log_interval == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for _, batch in enumerate(eval_loader):
                input_ids, labels = batch

                input_ids, labels = (
                    input_ids.to(train_args.device),
                    labels.to(train_args.device),
                )
                outputs = model(input_ids)

                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs.view(-1, tokenizer.vocab_size), labels.view(-1))
                eval_loss += loss.item()
        print(f"Epoch: {epoch}, Evaluation Loss: {eval_loss / len(eval_loader)}")

        # Save the trained model
        model_save_path = "llama2_darija"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")


def main(
    n_epochs: int = 10,
    train_data_path: str = settings.TRAIN_DATA_PATH,
    eval_data_path: str = settings.EVAL_DATA_PATH,
    tokenizer_name: str = settings.TOKENIZER_NAME,
):
    new_tokenizer = get_tokenizer(
        train_data_path=train_data_path, tokenizer_name=tokenizer_name
    )
    # Create dataset and dataloaders
    train_dataset = TextDataset(train_data_path, new_tokenizer, max_length=512)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    eval_dataset = TextDataset(eval_data_path, new_tokenizer, max_length=512)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    # Initialize training arguments
    train_args = TrainArgs(
        n_epochs=n_epochs,  # Number of epochs to train the model
        log_interval=10,  # How often to log the training progress
        lr=3e-4,  # Learning rate for the optimizer
        warmup_steps=4000,  # Number of warmup steps for the learning rate scheduler
        device="cuda" if torch.cuda.is_available() else "cpu",  # Compute device
        vocab_size=new_tokenizer.vocab_size,  # Size of the model's vocabulary
    )
    # Initialize llama arguments
    model_args = ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=256,
        vocab_size=new_tokenizer.vocab_size,
        ffn_dim_multiplier=4,
        norm_eps=1e-5,
        batch_size=32,
        max_seq_length=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pad_token_id=new_tokenizer.pad_token_id,  # Ensure this value is within the vocabulary size
    )

    # Initialize model
    model = Transformer(model_args)
    # Train the model
    train_model(model, train_loader, eval_loader, train_args, new_tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model.")
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=settings.TRAIN_DATA_PATH,
        help="Path to the training data",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=settings.EVAL_DATA_PATH,
        help="Path to the evaluation data",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=settings.TOKENIZER_NAME,
        help="Name of the tokenizer",
    )

    args = parser.parse_args()
    main(
        n_epochs=args.n_epochs,
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        tokenizer_name=args.tokenizer_name,
    )
