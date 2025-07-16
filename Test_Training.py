#Now we test the trained network
print("\nTrained Network Predictions:")
for i in range(len(training_input)):
    NN(training_input[i])
    predicted = NN.get_output_value()
    print(f"Input: {training_input[i]}, Actual: {actual_output[i]:.4f}, Predicted: {predicted.value:.4f}")
    print(f"Loss: {mean_squared_error([actual_output[i]], [predicted.value]):.6f}")
