import random
import math

class Neuron:

  #Constructor for this class
  def __init__(self,previous_neurons = (),input_layer = False):

    #Now the value of this Neuron was created based on the values of the direct inputs to this neuron
    #We create a set to avoid duplicates
    self.previous_neurons = set(previous_neurons)

    #We also associate weights with this neuron which is the same as the number of previous neurons
    self.weights = [Node(random.uniform(-1,1)) for _ in self.previous_neurons]

    #Each neuron has some bias associated with it
    self.bias = Node(random.uniform(-1,1))

    self.input_layer_neuron = input_layer

  #This is the function to print the node
  def __repr__(self):
    return f"Neuron(bias={self.bias})"

  #This function calculates the neuron_value based on the previous_neurons and weights and biases
  def calculate_neuron_value(self,input_value = 0):

    if self.input_layer_neuron:
      self.neuron_value = Node(input_value)
      return

    weighted_sum = 0
    for input,weight in zip(self.previous_neurons,self.weights):
      weighted_sum += input.neuron_value.value * weight.value

    #Each neuron has a value between -1 to 1 associated with it
    self.neuron_value = Node(weighted_sum + self.bias.value)

  #This function takes the neuron_value and inputs it into the activation function
  def activation_function(self):
    sigmoid_result = 1 / (1 + math.exp(-self.neuron_value.value))
    self.neuron_value = Node(sigmoid_result)


  #This function calculates the neuron value and performs the activation function on it
  def __call__(self,input = 0):
    self.calculate_neuron_value(input)
    self.activation_function()
    return self.neuron_value
