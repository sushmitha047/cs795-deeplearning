import os
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

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

def load_and_process_enwik8(file_path, sequence_length):
    with open(file_path, 'rb') as f:
        data = bytearray(f.read())
    data = np.array(data, dtype=np.uint8)
    
    num_sequences = len(data) // sequence_length
    data = data[:num_sequences * sequence_length]
    data = data.reshape((num_sequences, sequence_length))
    
    data = torch.tensor(data, dtype=torch.long)
    
    dataset = TensorDataset(data[:, :-1], data[:, 1:])  # input and target shifted by one position for prediction task
    return dataset


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
        
        # Compute sparse attention scores
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
            nn.Dropout(0.4),
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
            nn.Dropout(0.4),
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
    def __init__(self, embed_size, num_heads, num_layers, block_size, num_tokens, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embed_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, embed_size))
        self.enc_layers = nn.ModuleList([TransformerEncoderLayer(embed_size, num_heads, block_size) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([TransformerDecoderLayer(embed_size, num_heads, block_size) for _ in range(num_layers)])
        self.final_layer = nn.Linear(embed_size, num_tokens)

    def forward(self, src, tgt):
        assert src.size(0) == tgt.size(0), f"Source and target batch sizes do not match: {src.size(0)} != {tgt.size(0)}"
        src = self.embedding(src) + self.pos_embedding[:,:src.size(1),:]
        tgt = self.embedding(tgt) + self.pos_embedding[:,:tgt.size(1),:]
        memory = src
        for layer in self.enc_layers:
            memory = layer(memory)
        out = tgt
        for layer in self.dec_layers:
            out = layer(out, memory)
        out = self.final_layer(out)
        return out

    @property
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

running_loss = 0
running_total = 0

def train(model, criterion, optimizer, scheduler, data_loader, epoch, running_loss, running_total, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    start_time = time.time()

    for batch, (src, tgt) in enumerate(data_loader):
        optimizer.zero_grad()
        src = src.to(device)
        tgt = tgt.to(device)
        output = model(src, tgt[:, :-1])
        loss = criterion(output.view(-1, num_tokens), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(-1)
        correct += (predicted == tgt[:, 1:]).view(-1).sum().item()
        total += tgt[:, 1:].numel()

    scheduler.step()  # Update the learning rate

    running_loss += total_loss
    running_total += total

    avg_loss = total_loss / len(data_loader) # Average loss per token
    bits_per_dim = avg_loss / math.log(2)  # Convert to bits
    accuracy = correct / total
    end_time = time.time()
    epoch_duration = end_time - start_time

    print(f'\tTraining Loss: {total_loss/len(data_loader):.4f} | Accuracy: {accuracy:.4f} | Bits/Dim: {bits_per_dim:.2f} | Time: {epoch_duration:.2f}s')
    return total_loss / len(data_loader), accuracy

def validate(model, criterion, data_loader, running_loss, running_total):
    model.eval()  # Set the model to evaluation mode
    total_loss, total_correct, total_tokens = 0, 0, 0
    start_time = time.time()
    with torch.no_grad():  # No need to track gradients
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])  # Predict the next token
            loss = criterion(output.view(-1, num_tokens), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()

            # Calculate accuracy
            _, predictions = torch.max(output, dim=2)  # Get the index of the max log-probability
            correct = (predictions == tgt[:, 1:]).float().sum()  # Compare with ground truth
            total_correct += correct.item()
            total_tokens += tgt[:, 1:].numel()  # Total number of tokens
    running_loss += total_loss
    running_total += total_tokens
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_tokens
    
    bits_per_dim = avg_loss / math.log(2)
    accuracy = correct / total_tokens
    end_time = time.time()
    epoch_duration = end_time - start_time

    print(f'\tValidation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Bits/Dim: {bits_per_dim:.2f} | Time: {epoch_duration:.2f}s')
    return avg_loss, accuracy



def ensure_cpu_numpy(metric):
    if torch.is_tensor(metric):
        if metric.is_cuda:
            metric = metric.cpu()
        metric = metric.numpy()
    elif isinstance(metric, list):
        return [ensure_cpu_numpy(m) for m in metric]
    return metric

def plot_metrics(metrics, labels, title, filename):
    epochs = range(1, len(metrics[0]) + 1)
    plt.figure(figsize=(10, 5))
    processed_metrics = [ensure_cpu_numpy(metric) for metric in metrics]

    for metric, label in zip(processed_metrics, labels):
        plt.plot(epochs, metric, label=label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(filename)
    plt.close()


sequence_length = 256 
file_path = './data/enwik8'

# Load and process dataset
dataset = load_and_process_enwik8(file_path, sequence_length)

print(f"total length of dataset = {len(dataset)}")

# Split dataset into training, validation, and testing
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

print(f"train size = {train_size}")
print(f"valid size = {val_size}")
print(f"testi size = {test_size}")

# Create DataLoader instances for each set
batch_size = 768
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

print(f"test dataset size = {len(test_dataset)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Parameters
embed_size = 256
num_heads = 8
epochs = 5
block_size = 8
num_layers = 12 
# original = 30
num_tokens = 256  
max_seq_length = sequence_length 

# Instantiate the Model
model = Transformer(embed_size, num_heads, num_layers, block_size, num_tokens, max_seq_length).to(device)
print(f"total number of trainable parameters in model: {model.total_parameters}")
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]) # Distribute across 4 GPUs

# Optimizer and Loss Function
# prev_lr = 0.0003
weight_decay = 0.01
optimizer = Adam(model.parameters(), lr=0.00035, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()

#epochs = 5  # Define the number of epochs

losses, accuracies, val_losses, val_accuracies = [], [], [], []
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}")
    train_loss, train_accuracy = train(model, criterion, optimizer, scheduler, train_dataloader, epoch, running_loss, running_total, device)# reset_accumulators=True)#, embed_size,device)
    val_loss, val_accuracy = validate(model, criterion, val_dataloader, running_loss, running_total)
    # Save metrics
    losses.append(train_loss)
    accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

running_loss = 0
running_total = 0

test_loss, test_accuracy = validate(model, criterion, test_dataloader, running_loss, running_total)
print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}')
 
plot_metrics([losses, val_losses], ['Training Loss', 'Validation Loss'], 'Training and Validation Loss', 'loss_plot_text.png')
plot_metrics([accuracies, val_accuracies], ['Training Accuracy', 'Validation Accuracy'], 'Training and Validation Accuracy', 'accuracy_plot_text.png')


