from load_data import load_data
from baseline import AE
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn
import numpy as np
import os


base_dir = os.getcwd()
study = 'dataset'
ttp = 12
ttp_p = 11
gene_cap = 2000
cell_cap = 2000
dense_mat_list, _ = load_data(base_dir, study, ttp, gene_cap, cell_cap)
print(len(dense_mat_list))



# Convert tensor
input_tensor = torch.tensor(np.concatenate(dense_mat_list), dtype=torch.float32)

# Shuffle 
input_tensor = input_tensor[torch.randperm(input_tensor.size()[0])]

# Split
train_size = int(0.7 * len(input_tensor))
valid_size = int(0.15 * len(input_tensor))
test_size = len(input_tensor) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(input_tensor, [train_size, valid_size, test_size])

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Initialization
[cell_num, gene_num] = input_tensor.size()

ae = AE(cell_num=cell_num, gene_num=gene_num, latent_space=64)

# Recon
criterion = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=0.001)

# Train
def train(model, data_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in data_loader:
            inputs = data.unsqueeze(1)  # Add a channel dimension
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}')

# Eval
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            inputs = data.unsqueeze(1)  # Add a channel dimension
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Train
train(ae, train_loader, optimizer, criterion, epochs=10)

latent = ae.latent(input_tensor)
print(f"Latent representation: {latent.size()}")
# Evaluating the model
valid_loss = evaluate(ae, valid_loader, criterion)
test_loss = evaluate(ae, test_loader, criterion)
print(f'Validation Loss: {valid_loss}, Test Loss: {test_loss}')
