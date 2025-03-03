import torch
import torch.nn as nn
import torch.optim as optim
from cnn import CNN
from dataset import get_dataloaders

def train_model(epochs=50, batch_size=64, checkpoint_path='./model_checkpoint.pth'):
    device = torch.device('cpu')

    train_loader, _ = get_dataloaders(batch_size)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
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
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

train_model()