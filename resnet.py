import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import copy

# ✅ 1. 디바이스 설정: MPS (Apple Silicon GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ 2. 모델 준비: ResNet18 (Pretrained 선택 가능)
model = models.resnet18(pretrained=False).to(device)
model.train()

# ✅ 3. 데이터셋 (CIFAR10 원본)
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ✅ 4. 여러 레이어의 feature를 저장할 hook 등록
features = {}

def get_hook(name):
    def hook(module, input, output):
        features[name] = output.detach().cpu()  # CPU로 옮겨서 저장
    return hook

# 추적하고 싶은 레이어들: conv1, layer1~layer4
target_layers = {
    "conv1": model.conv1,
    "layer1": model.layer1,
    "layer2": model.layer2,
    "layer3": model.layer3,
    "layer4": model.layer4
}

hook_handles = []
for name, layer in target_layers.items():
    hook_handles.append(layer.register_forward_hook(get_hook(name)))

# ✅ 5. 로그 저장용 리스트
feature_logs = {name: [] for name in target_layers.keys()}

# ✅ 6. 학습 루프
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_iterations = 10

for i, (x, _) in enumerate(dataloader):
    if i >= num_iterations:
        break

    x = x.to(device)

    # forward pass → hook이 자동으로 feature 저장
    _ = model(x)

    # 각 레이어 feature 저장
    for name in features:
        feature_logs[name].append(copy.deepcopy(features[name]))

    # backward & optimizer step (더미 학습)
    output = model(x)
    loss = output.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("✅ 실험 완료. feature_logs에 각 레이어 출력이 저장되었습니다.")

# ✅ 7. hook 제거 (필수는 아니지만 깔끔하게)
for h in hook_handles:
    h.remove()
