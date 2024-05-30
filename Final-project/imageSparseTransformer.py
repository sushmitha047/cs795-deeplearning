import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
#from torchvision.models import resnet18  # Using ResNet18 as a feature extractor
import numpy as np

num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

print("Is CUDA available:", torch.cuda.is_available())
print("Number of GPU(s):", torch.cuda.device_count())

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(current_device)
    print(f"Device Name: {gpu_properties.name}")
    print(f"Number of Streaming Multiprocessors (SMs): {gpu_properties.multi_processor_count}")
else:
    print("No GPU available.")


class SparseAttention(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super(SparseAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size, "Embed size must be divisible by num_heads"

        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)

        self.feature_projection = nn.Parameter(torch.randn(self.head_dim, self.head_dim))

    def forward(self, x):
        B, N, E = x.shape
        q = self.queries(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.keys(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.values(x).reshape(B, N, self.num_heads, self.head_dim)

        # Projecting to feature space
        q = torch.einsum('bnhe,ei->bnhi', q, self.feature_projection)
        k = torch.einsum('bnhe,ei->bnhi', k, self.feature_projection)

        # Compute attention scores
        #scores = torch.einsum('bnhi,bmhi->bnhm', q, k) / math.sqrt(self.head_dim)
        # Sparse attention
        scores = torch.full((B, N, self.num_heads, N), float('-inf'), device=x.device)
        for i in range(0, N, self.block_size):
            scores[:, i:i+self.block_size, :, i:i+self.block_size] = torch.einsum('bnhd,bmhd->bnhm', q[:, i:i+self.block_size], k[:, i:i+self.block_size]) / math.sqrt(self.head_dim)

        attention = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.einsum('bnhm,bmhe->bnhe', attention, v).reshape(B, N, E)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        self.sparse_attention = SparseAttention(embed_size, num_heads, block_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(4 * embed_size, embed_size)
        )

    def forward(self, src):
        src = src + self.sparse_attention(self.norm1(src))
        src = src + self.ff(self.norm2(src))
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        self.sparse_attention = SparseAttention(embed_size, num_heads, block_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(4 * embed_size, embed_size)
        )

    def forward(self, tgt, memory):
        if memory.size(0) > tgt.size(0):
            memory = memory[:tgt.size(0), :, :]
        # Ensure tgt and memory are compatible
        if tgt.size(1) != memory.size(1):
            padding = torch.zeros_like(memory[:, :(tgt.size(1) - memory.size(1)), :])
            memory = torch.cat([memory, padding], dim=1)
        tgt_attention = self.sparse_attention(self.norm1(tgt))
        memory_attention = self.sparse_attention(self.norm2(memory[:, :tgt.size(1), :]))
        tgt = tgt + tgt_attention + memory_attention
        tgt = tgt + self.ff(self.norm3(tgt))
        return tgt

class Transformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_size, num_heads, num_layers, block_size, num_classes, max_seq_length):
        super().__init__()
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_size, img_size)
        num_patches = self.patch_embedding.num_patches
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        
        # Transformer layers
        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(embed_size, num_heads, block_size) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([TransformerDecoderLayer(embed_size, num_heads, block_size) for _ in range(num_layers)])
        
        # Output layer
        self.final_layer = nn.Linear(embed_size, num_classes)

        #print(f"Number of patches (num_patches): {num_patches}")
        #print(f"Max sequence length (max_seq_length): {max_seq_length}")


    def forward(self, src):
        B = src.size(0)
        src = self.patch_embedding(src)
        
        # Prepend the cls token to the input
        cls_tokens = self.cls_token.expand(B, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)
        
        #print(f"src size: {src.size()}")
        #print(f"pos_embedding size: {self.pos_embedding.size()}")
        # Add positional embeddings
        src += self.pos_embedding
        
        # Transformer Encoder
        for layer in self.enc_layers:
            src = layer(src)
        
        out = self.final_layer(src[:, 0])
        
        return out

    @property
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        #self.num_patches = (img_size // patch_size) ** 2
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.projection = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # [B, embed_size, num_patches**0.5, num_patches**0.5]
        x = x.flatten(2)  # Flatten patches into a single dimension
        x = x.transpose(1, 2)  # [B, num_patches, embed_size]
        #print(f"After PatchEmbedding: {x.size()}")
        return x

def validate(model, criterion, data_loader, num_classes):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predictions = torch.max(outputs, dim=1)
            total_correct += (predictions == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total
    bits_per_dim = avg_loss / math.log(2)
    print(f'\tValidation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Bits/Dim: {bits_per_dim:.4f}')
    return total_loss / len(data_loader), accuracy

def train(model, criterion, optimizer, scheduler, data_loader, epoch, num_classes, running_loss, running_total):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    start_time = time.time()

    for images, targets in data_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        #print(f"Input size to model: {images.size()}")
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == targets).sum().item()
        total += targets.size(0)
    scheduler.step()

    running_loss += total_loss
    running_total += total

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total
    bits_per_dim = avg_loss / math.log(2)
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f'Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Bits/Dim: {bits_per_dim:.4f}, Time: {epoch_duration:.2f}s')
    return total_loss/len(data_loader), accuracy


def plot_metrics(metrics, labels, title, filename):
    epochs = range(1, len(metrics[0]) + 1)
    plt.figure(figsize=(10, 5))
    # Ensure all metrics are numpy arrays, handling tensors if present
    processed_metrics = []
    for metric in metrics:
        if torch.is_tensor(metric):
            # Check if the tensor is on CUDA and move to CPU
            if metric.is_cuda:
                metric = metric.cpu()
            metric = metric.numpy()
        processed_metrics.append(metric)

    for metric, label in zip(processed_metrics, labels):
        # Debugging output to verify data types
        #print(f"Type of metric before plotting: {type(metric)}")
        plt.plot(epochs, metric, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(filename)
    plt.close()

# CIFAR-10 Data Preparation
transform = transforms.Compose([
    #transforms.Resize((224, 224)),  # Resize images to fit ResNet18 input
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print(f"num of training sapmles: {len(cifar_train)}")
print(f"num of test smackles: {len(cifar_test)}")

batch_size = 950

# CIFAR-10 images are 32x32 pixels with 3 color channels
img_size = 32 
patch_size = 8
in_channels = 3  # images are RGB
embed_size = 256
num_heads = 2
num_layers = 128
block_size = 8 
num_classes = 10
max_seq_length = (img_size // patch_size) ** 2 + 1  # Including class token

train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)

print(f"num of train loader = {len(train_loader)}")
print(f"num of test laoder  = {len(test_loader)}")

# Initialize Model, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")

num_epochs = 15
running_loss = 0
running_total = 0

losses, accuracies, val_losses, val_accuracies = [], [], [], []

block_sizes = [8]#, 16, 32, 64]  # Example block sizes to test
results = {}

for block in block_sizes:
    print(f"block size = {block}")
    model = Transformer(img_size, patch_size, in_channels, embed_size, num_heads, num_layers, block_size, num_classes, max_seq_length).to(device)

    print(f"total number of trainable parameters in model: {model.total_parameters}")
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]) # Distribute across 2 GPUs

    weight_decay = 0.001

    # Optimizer and Loss Function
    optimizer = Adam(model.parameters(), lr=0.00035, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, num_epochs + 1):
        loss, acc = train(model, criterion, optimizer,scheduler, train_loader, epoch, num_classes, running_loss, running_total)
        val_loss, val_acc = validate(model, criterion, test_loader, num_classes)
        # Save metrics
        losses.append(loss)
        accuracies.append(acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        #train_loss, validation_metrics = train_and_evaluate(model)
        results[block] = (loss, acc)


running_loss = 0
running_total = 0

test_loss, test_accuracy = validate(model, criterion, test_loader, num_classes)# running_loss, running_total)
print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}')

plot_metrics([losses,val_losses],['Training Loss', 'Validation Loss'],'Training and Validation Loss', 'loss_plot_img.png' )
plot_metrics([accuracies,val_accuracies], ['Training Accuracy','Validation Accuracy'], 'Training and Validation Accuracy', 'accurayc_plot_img.png')

