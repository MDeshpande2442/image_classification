import torch
import torchvision
from torchvision import transforms
from mnist_model import MNISTModel

def main():
    # Load test data
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # Load model
    model = MNISTModel()
    model.load_state_dict(torch.load('models/mnist_model.pth'))
    model.eval()

    # Evaluate
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    main()
