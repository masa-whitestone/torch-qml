"""
Hybrid Quantum Neural Network for MNIST Classification

Example of torchqml usage matching the user's design goals.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchqml as tq
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# ==================== 設定 ====================

torch.manual_seed(22)
try:
    tq.set_random_seed(44)
except:
    pass

# GPU使用 (Environment check)
try:
    tq.set_backend("nvidia")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Backend: nvidia, Device: {device}")
except:
    tq.set_backend("qpp-cpu")
    device = torch.device("cpu")
    print(f"Backend: qpp-cpu, Device: {device}")

# ==================== データ準備 ====================

def prepare_data(target_digits, sample_count, test_size):
    """MNISTデータセットを準備"""
    # Check if data exists or download
    os.makedirs("./data", exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    except Exception as e:
        print(f"Failed to download MNIST: {e}")
        # Create dummy data for verification if download fails
        print("Using dummy data")
        x = torch.randn(sample_count * 2, 28, 28)
        y = torch.randint(0, 2, (sample_count * 2,)).float()
        idx = torch.randperm(sample_count * 2)[:sample_count]
        return train_test_split(x[idx].unsqueeze(1).to(device), y[idx].to(device), test_size=test_size/100, shuffle=True, random_state=42)

    # 指定した数字のみフィルタ
    idx = (dataset.targets == target_digits[0]) | (dataset.targets == target_digits[1])
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    
    # サブセットを選択
    subset_indices = torch.randperm(dataset.data.size(0))[:sample_count]
    x = dataset.data[subset_indices].float().unsqueeze(1).to(device)
    y = dataset.targets[subset_indices].float().to(device)
    
    # ラベルを0/1に変換
    y = torch.where(y == min(target_digits), 0.0, 1.0)
    
    return train_test_split(x, y, test_size=test_size/100, shuffle=True, random_state=42)

# ==================== モデル定義 ====================

class QuantumLayer(tq.QuantumModule):
    """量子層"""
    
    def __init__(self):
        super().__init__()
        # パラメータは古典NNから受け取るので、ここでは定義不要
    
    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Args:
            thetas: [batch_size, 2] - 2つの回転パラメータ
        
        Returns:
            期待値: [batch_size]
        """
        batch_size = thetas.shape[0]
        q = tq.qvector(1, batch_size, device=thetas.device)
        
        # 量子回路
        tq.ry(thetas[:, 0], q[0])
        tq.rx(thetas[:, 1], q[0])
        
        # Z期待値を測定
        return tq.expval(q, tq.Z(0))


class HybridQNN(nn.Module):
    """ハイブリッド量子古典ニューラルネットワーク"""
    
    def __init__(self):
        super().__init__()
        
        # 古典層
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.25)
        
        # 量子層
        self.quantum = QuantumLayer()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 画像をフラット化
        x = x.view(-1, 28 * 28)
        
        # 古典層
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.dropout(x)
        
        # 量子層 + sigmoid
        x = torch.sigmoid(self.quantum(x))
        
        return x


# ==================== 訓練 ====================

def train():
    # ハイパーパラメータ
    sample_count = 100 # Reduced for quick check
    target_digits = [5, 6]
    test_size = 30
    epochs = 5 # Reduced for quick check
    classification_threshold = 0.5
    
    print("Preparing data...")
    # データ準備
    x_train, x_test, y_train, y_test = prepare_data(target_digits, sample_count, test_size)
    
    print("Initializing model...")
    # モデル
    model = HybridQNN().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.001, weight_decay=0.8)
    loss_fn = nn.BCELoss()
    
    # 訓練履歴
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": []
    }
    
    def accuracy(y_true, y_pred):
        return ((y_pred >= classification_threshold) == y_true).float().mean().item()
    
    print("Starting training loop...")
    # 訓練ループ
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        y_pred_train = model(x_train)
        train_loss = loss_fn(y_pred_train, y_train)
        train_loss.backward()
        optimizer.step()
        
        train_acc = accuracy(y_train, y_pred_train)
        history["train_loss"].append(train_loss.item())
        history["train_acc"].append(train_acc)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred_test = model(x_test)
            test_loss = loss_fn(y_pred_test, y_test)
            test_acc = accuracy(y_test, y_pred_test)
        
        history["test_loss"].append(test_loss.item())
        history["test_acc"].append(test_acc)
        
        print(f"Epoch {epoch}: Train Loss={train_loss.item():.4f}, "
              f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    return history

if __name__ == "__main__":
    try:
        history = train()
        print("HQNN Example Run Successful")
    except Exception as e:
        print(f"HQNN Example Run Failed: {e}")
        import traceback
        traceback.print_exc()
