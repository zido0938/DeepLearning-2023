import argparse
import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
from model import CustomResNet
import torch
from torch.optim.lr_scheduler import StepLR

def acc(pred, label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Add a learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(args.epochs):
        train_losses = []
        train_acc = 0.0
        total = 0
        print(f"[Epoch {epoch+1} / {args.epochs}]")

        model.train()
        pbar = tqdm(data_loader)
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)
            optimizer.zero_grad()

            output = model(image)

            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        # Adjust learning rate with scheduler
        scheduler.step()

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc / total

        print(f'Epoch {epoch+1}')
        print(f'train_loss : {epoch_train_loss}')
        print(f'train_accuracy : {epoch_train_acc * 100:.3f}%')
        print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]}')

        torch.save(model.state_dict(), f'{args.save_path}/model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    args.epochs = 12  
    args.batch_size = 64
    args.learning_rate = 0.0001

    print("==============================")
    print("Save path:", args.save_path)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())

    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    print("==============================")

    train_loader, _ = make_data_loader(args)
    model = CustomResNet(num_classes=10)
    model.to(device)

    train(args, train_loader, model)