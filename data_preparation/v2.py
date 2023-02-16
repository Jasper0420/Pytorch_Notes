
torch.manual_seed(13)

# Builds tensors from numpy arrays BEFORE split
x_tensor = torch.as_tensor(x).float()                          # 1)
y_tensor = torch.as_tensor(y).float()                          # 1)

# Builds dataset containing ALL data points
dataset = TensorDataset(x_tensor, y_tensor)

# Performs the split
ratio = .8#训练集占比
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train
train_data, val_data = random_split(dataset, [n_train, n_val])  # 2) 

# Builds a loader of each set
train_loader = DataLoader(
    dataset=train_data, 
    batch_size=16, 
    shuffle=True,
)
val_loader = DataLoader(dataset=val_data, batch_size=16)        # 3)
