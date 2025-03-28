import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# 1. 데이터셋 로드 (CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# 2. ResNet 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10에 맞게 마지막 FC Layer 변경
model = model.to(device)

# 3. Hidden Layer 값을 저장하기 위한 Hook 설정
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# 특정 layer에 hook 등록 (예: layer1, layer2)
model.layer1[0].register_forward_hook(get_activation('layer1'))
model.layer2[0].register_forward_hook(get_activation('layer2'))

# 4. 학습 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 모델 학습
epochs = 1  # 빠른 실험을 위해 1 epoch만 실행
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.4f}")
            running_loss = 0.0

print("Finished Training")

# 6. Hidden Layer Activation 확인
model.eval()
sample_data, _ = next(iter(testloader))
sample_data = sample_data.to(device)

with torch.no_grad():
    _ = model(sample_data)

layer1_activation = activation['layer1'].cpu().numpy()
layer2_activation = activation['layer2'].cpu().numpy()

# 7. Feature Map 시각화 (첫 번째 샘플 기준)
plt.figure(figsize=(10, 5))
for i in range(8):  # 첫 번째 채널 8개만 출력
    plt.subplot(2, 4, i+1)
    plt.imshow(layer1_activation[0, i, :, :], cmap='viridis')
    plt.axis('off')
plt.suptitle("Layer 1 Feature Maps")
plt.show()
