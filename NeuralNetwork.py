#Now we define the Neural Network class
class NeuralNetwork:

  def __init__(self, layer_sizes):

    if len(layer_sizes) < 2:
            raise ValueError("Neural network must have at least input and output layers")

    #We pass in an array which indicates the size of each layer of the neural network
    self.layer_sizes = layer_sizes

    #This contains all the layers in the NN
    self.layers = []

    #Creating input layer
    input_layer = Layer(layer_sizes[0], input_layer=True)
    self.layers.append(input_layer)

    #Creating the hidden layers
    for i in range(1, len(layer_sizes) - 1):
      hidden_layer = Layer(layer_sizes[i], previous_layer=self.layers[-1])
      self.layers.append(hidden_layer)

    #Creating the output layer
    output_layer = Layer(layer_sizes[-1], previous_layer=self.layers[-1], output_layer = True)
    self.layers.append(output_layer)

  def get_parameters(self):
    parameters = []
    for layer in self.layers:
      parameters += layer.get_parameters()
    return parameters

  #Now we do a forward pass to find the value of all the neurons in the network
  def forward_pass(self,inputs):

    if(len(inputs) != self.layer_sizes[0]):
      raise ValueError("Number of inputs must match the number of neurons in the input layer")

    #Finding the values of the neurons for the input layer first
    current_outputs = self.layers[0](inputs)

    #Passing through all subsequent layers
    for layer in self.layers[1:]:
        current_outputs = layer()

    return current_outputs

  #Now we do backpropogation through the neural network


  def __repr__(self):
    return f"NeuralNetwork(layers={self.layer_sizes})"

  def __call__(self,inputs):
    return self.forward_pass(inputs)

  def get_output_value(self):
        """Get the numeric value from the output layer"""
        output_layer = self.layers[-1]

        if len(output_layer.neurons) == 1:
          return output_layer.neurons[0].neuron_value
        else:
          return [neuron.neuron_value for neuron in output_layer.neurons]
