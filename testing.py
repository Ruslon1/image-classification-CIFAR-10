import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
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


def test_custom_image(image_path, checkpoint_path='./model_checkpoint.pth'):
    device = torch.device('cpu')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Display the image going into the model
    # De-normalize the image for display
    denormalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )
    image_for_display = denormalize(image[0]).clamp(0, 1).permute(1, 2, 0)  # Convert to HWC format

    plt.imshow(image_for_display.cpu().numpy())
    plt.title("Image Going Into Neural Network")
    plt.axis('off')
    plt.show()

    model = CNN().to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    predicted_class = class_names[predicted.item()]
    print(f"Predicted Class: {predicted_class}")

#test_model()
test_custom_image('./image.jpg', checkpoint_path="./new_checkpoint")