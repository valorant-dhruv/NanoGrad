#Validating the neural network
print("Validating the neural network...")

#Validation data with different inputs
validation_input = [[4, 25000],
                    [2, 12000],
                    [6, 150000],
                    [3, 75000],
                    [1, 5000],
                    [8, 200000]]

validation_input = normalise_input(validation_input)
validation_actual_output = [38.5, 22.3, 142.8, 67.2, 15.9, 175.6]
validation_actual_output = normalise_output(validation_actual_output)

print("=" * 50)

#Now we do a forward pass and find the predicted output
validation_predicted_output = []
for i in range(len(validation_input)):
  NN(validation_input[i])
  validation_predicted_output.append(NN.get_output_value())

#Now we calculate the loss from the predicted output
validation_loss = mean_squared_error(validation_actual_output, validation_predicted_output)

print("=" * 50)
print("Validation completed!")
print(f"Validation Loss: {validation_loss.value:.6f}")

# Print comparison of actual vs predicted
print("\nValidation Results:")
print("Actual vs Predicted:")
for i, (actual, predicted) in enumerate(zip(validation_actual_output, validation_predicted_output)):
    print(f"Sample {i+1}: Actual={actual:.4f}, Predicted={predicted.value:.4f}")
