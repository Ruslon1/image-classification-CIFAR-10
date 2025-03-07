import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from cnn import CNN
from dataset import get_dataloaders

def train_model(batch_size=2048, checkpoint_path='./model_checkpoint.pth'):
    device = torch.device('cuda')

    train_loader, _ = get_dataloaders(batch_size)
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0


    for epoch in range(start_epoch, start_epoch + 300):
        start_time = time.time()

        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch [{epoch+1}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f} seconds")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': running_loss
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
if __name__ == "__main__":
    train_model(checkpoint_path="./new_checkpoint")