def print_network_structure(neural_network):
    print("Neural Network Structure:")
    print("=" * 40)
    for i, layer in enumerate(neural_network.layers):
      if i == 0:
        layer_type = "Input"
      elif i == len(neural_network.layers) - 1:
        layer_type = "Output"
      else:
        layer_type = f"Hidden {i}"
      print(f"{layer_type} Layer: {layer.number_of_neurons} neurons")
            
      # Print neuron details
      for j, neuron in enumerate(layer.neurons):
        print(f"  Neuron {j}: bias={neuron.bias.value:.3f}, weights={[w.value for w in neuron.weights]}, value={neuron.neuron_value.value:.3f}")
      print("=" * 40)
