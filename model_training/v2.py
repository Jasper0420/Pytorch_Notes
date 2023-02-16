
# Defines number of epochs
n_epochs = 1000

losses = []

# For each epoch...
for epoch in range(n_epochs):
    # inner loop
    mini_batch_losses = []                              # 4)
    for x_batch, y_batch in train_loader:               # 1)
        # the dataset "lives" in the CPU, so do our mini-batches
        # therefore, we need to send those mini-batches to the
        # device where the model "lives"
        x_batch = x_batch.to(device)                    # 2)
        y_batch = y_batch.to(device)                    # 2)

        # Performs one train step and returns the 
        # corresponding loss for this mini-batch
        mini_batch_loss = train_step(x_batch, y_batch)  # 3)
        mini_batch_losses.append(mini_batch_loss)       # 4)

    # Computes average loss over all mini-batches
    # That's the epoch loss
    loss = np.mean(mini_batch_losses)                   # 5)    
    losses.append(loss)
