import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.onnx
import os

# 데이터 다운로드
datasets.MNIST.mirrors = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/"
]

class SmallFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
    def forward(self, x):
        return self.net(x)

transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

model = SmallFC()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    for x, y in loader:
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

os.makedirs("models", exist_ok=True)
model.eval()
dummy = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model, dummy, "models/mnist_fc.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=11)
print("Saved: models/mnist_fc.onnx")