import torch
from cnn import CNN
from dataset import get_dataloaders

def test_model(checkpoint_path='./model_checkpoint.pth', batch_size=64):
    device = torch.device('cpu')

    _, test_loader = get_dataloaders(batch_size)

    model = CNN().to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
