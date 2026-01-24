import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from h2q_project.models.simple_lstm import SimpleLSTM
from h2q_project.data_loader.data_loader import TextDataset, create_vocab, preprocess_data
from h2q_project.utils.arguments import parse_args


def train_epoch(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Data preparation
    preprocess_data(args.train_file)
    vocab = create_vocab([args.train_file])
    train_data = TextDataset(args.train_file, vocab, args.max_length)
    val_data = TextDataset(args.val_file, vocab, args.max_length) if args.val_file else None

    train_iterator = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_iterator = DataLoader(val_data, batch_size=args.batch_size) if val_data else None

    # Model and optimizer initialization
    model = SimpleLSTM(len(vocab), args.embedding_dim, args.hidden_dim, args.num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, args.clip)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}')

        if val_iterator:
            valid_loss = evaluate(model, val_iterator, criterion)
            print(f'\t Val. Loss: {valid_loss:.3f}')

    # Save the model
    torch.save(model.state_dict(), args.save_path)
    print(f'Model saved to {args.save_path}')


if __name__ == '__main__':
    main()