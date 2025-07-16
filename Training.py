#Training the neural network
print("Training the neural network...")

#Let's assume the number of parameters are between
training_input = [[3, 15000],
                  [1, 2500],
                  [7, 180000],
                  [2, 45000],
                  [5, 320000],
                  [1, 8000]]

training_input = normalise_input(training_input)
actual_output = [42.3, 18.7, 156.2, 51.8, 198.4, 25.1]
actual_output = normalise_output(actual_output)

print("=" * 50)

learning_rate = 0.01

epochs = 1000

NN = NeuralNetwork([2,4,1])

for epoch in range(epochs):

  #Now we do a forward pass and find the predicted output
  predicted_output = []
  for i in range(len(training_input)):
    NN(training_input[i])
    predicted_output.append(NN.get_output_value())

  #Now we calculate the loss from the predicted output
  loss = mean_squared_error(actual_output, predicted_output)

  #Now we get the parameters
  parameters = NN.get_parameters()

  #Zero the gradients of all the parameters
  for param in parameters:
        param.gradient = 0.0

  #Do a backward pass from the loss function
  loss.backward()

  # Update parameters using gradient descent
  for param in parameters:
        param.value -= learning_rate * param.gradient

  # Print progress every 50 epochs
  if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.value:.6f}")

print("=" * 50)
print("Training completed!")
print(f"Final Loss: {loss.value:.6f}")
