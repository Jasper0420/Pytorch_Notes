
# Defines number of epochs
n_epochs = 200

losses = []
val_losses = []                                             # 3)

for epoch in range(n_epochs):
    # inner loop
    loss = mini_batch(device, train_loader, train_step)
    losses.append(loss)
    
    # VALIDATION - no gradients in validation!
    with torch.no_grad():                                    # 1)
        val_loss = mini_batch(device, val_loader, val_step)  # 2)
        val_losses.append(val_loss)                          # 3)    
