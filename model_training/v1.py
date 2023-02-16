
# Defines number of epochs
n_epochs = 1000

losses = []                                            # 2)

# For each epoch...
for epoch in range(n_epochs):
    # Performs one train step and returns the corresponding loss
    loss = train_step(x_train_tensor, y_train_tensor)  # 1)
    losses.append(loss)                                # 2)
