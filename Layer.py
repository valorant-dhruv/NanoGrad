#Each layer has multiple neurons associated with it
#There are essentially three types of layers in the neural network
#The input layer, hidden layers and the output layer
class Layer:

  def __init__(self, number_of_neurons, previous_layer=None, input_layer=False, output_layer = False):
    self.number_of_neurons = number_of_neurons
    self.previous_layer = previous_layer
    self.input_layer = input_layer
    self.output_layer = output_layer
    self.neurons = []

    #Now we add the neurons to this layer
    for i in range(self.number_of_neurons):

      #If it is the input layer then we pass the flag accordingly
      if self.input_layer:
        self.neurons.append(Neuron(input_layer = True))
      elif self.output_layer:
        self.neurons.append(Neuron(previous_layer.neurons, output_layer = True))
      else:
        self.neurons.append(Neuron(previous_layer.neurons))

  def __repr__(self):
    return f"Layer(neurons_count = {self.number_of_neurons}, neurons={self.neurons})"

  def get_parameters(self):
    parameters = []
    for neuron in self.neurons:
      parameters += neuron.get_parameter()
    return parameters

  #This function assigns values to the neurons of the layer
  #If it is an input layer, we assign the value that was passed to the given function
  #Otherwise we calculate the values of each neuron in the layer based on the previous neurons
  def __call__(self, inputs=None):
    outputs = []
    if self.input_layer:
      for neuron, input_value in zip(self.neurons, inputs):
        outputs.append(neuron(input_value))

    else:
      for neuron in self.neurons:
        outputs.append(neuron())

    return outputs
