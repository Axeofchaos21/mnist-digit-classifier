import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Set hyperparameters & computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 1e-3
epoch_num = 5

# 2. Data preprocessing pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load dataset
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. Define fully-connected MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平张量
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out

# 4. Initialize model, loss function and optimizer
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. Training loop
train_loss_list = []
train_acc_list = []

for epoch in range(epoch_num):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (pred == label).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_loss_list.append(epoch_loss)
    train_acc_list.append(epoch_acc)
    print(f"Epoch [{epoch+1}/{epoch_num}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

# 6. Testing evaluation flow
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        outputs = model(data)
        _, pred = torch.max(outputs.data, 1)
        test_total += label.size(0)
        test_correct += (pred == label).sum().item()

test_acc = 100 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")

# 7. Plot training loss and accuracy curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list)
plt.title("Train Loss")
plt.xlabel("Epoch")

plt.subplot(1, 2, 2)
plt.plot(train_acc_list)
plt.title("Train Accuracy")
plt.xlabel("Epoch")
plt.show()

# Save trained model weights
torch.save(model.state_dict(), "mnist_mlp.pth")
print("Model saved as mnist_mlp.pth")
