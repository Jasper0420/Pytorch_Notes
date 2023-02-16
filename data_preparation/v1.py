
# Our data was in Numpy arrays, but we need to transform them
# into PyTorch's Tensors
x_train_tensor = torch.as_tensor(x_train).float()
y_train_tensor = torch.as_tensor(y_train).float()

# Builds Dataset
train_data = TensorDataset(x_train_tensor, y_train_tensor)  # 1)

# Builds DataLoader
train_loader = DataLoader(                                  # 2)
    dataset=train_data, 
    batch_size=16, 
    shuffle=True,
)
