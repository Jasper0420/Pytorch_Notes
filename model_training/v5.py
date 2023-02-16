
# Defines number of epochs
n_epochs = 500

losses = []
val_losses = []

for epoch in range(n_epochs):
    # inner loop
    loss = mini_batch(device, train_loader, train_step)
    losses.append(loss)
    
    # VALIDATION - no gradients in validation!
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step)
        val_losses.append(val_loss)
    
    # Records both losses for each epoch under tag "loss"
    writer.add_scalars(main_tag='loss',      # 1)
                       tag_scalar_dict={
                            'training': loss, 
                            'validation': val_loss},
                       global_step=epoch)

# Closes the writer
writer.close()
