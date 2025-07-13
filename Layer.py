#Each layer has multiple neurons associated with it
class Layer:

  def __init__(self, number_of_neurons, previous_layer=None, input_layer=False):
    self.number_of_neurons = number_of_neurons
    self.previous_layer = previous_layer
    self.input_layer = input_layer
    self.neurons = []

    #Now we add the neurons to this layer
    for i in range(self.number_of_neurons):

      #If it is the input layer then we pass the flag accordingly
      if self.input_layer:
        self.neurons.append(Neuron(input_layer = True))
      else:
        self.neurons.append(Neuron(previous_layer.neurons))    

  def __repr__(self):
    return f"Layer(neurons={self.neurons})"
  
  #This function assigns values to neurons in the given layer
  def __call__(self, inputs=None):
    outputs = []
    if self.input_layer:
      for neuron, input_value in zip(self.neurons, inputs):
        outputs.append(neuron(input_value))

    else:
      for neuron in self.neurons:
        outputs.append(neuron())

    return outputs
