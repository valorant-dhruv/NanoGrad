import random
import math

class Neuron:

  #Constructor for this class
  def __init__(self,previous_neurons = (),input_layer = False, output_layer = False):

    #Now the value of this Neuron was created based on the values of the direct inputs to this neuron
    #We create a set to avoid duplicates
    self.previous_neurons = set(previous_neurons)

    #We also associate weights with this neuron which is the same as the number of previous neurons
    #self.weights = [Node(random.uniform(-1,1)) for _ in self.previous_neurons]
    fan_in = len(previous_neurons)
    if input_layer:
      fan_in = 1
    limit = (6.0 / fan_in) ** 0.5  # Xavier initialization
    self.weights = [Node(random.uniform(-limit, limit)) for _ in previous_neurons]

    #Each neuron has some bias associated with it
    #The input layer neurons don't have any bias associated with them
    if not input_layer:
      #self.bias = Node(random.uniform(-1,1))
      self.bias = Node(0.0)

    #This flag determines if the neuron is an input layer neuron or not
    self.input_layer_neuron = input_layer
    self.output_layer = output_layer

  #This is the function to print the Neuron
  def __repr__(self):
    return f"Neuron(bias={self.bias})"

  def sigmoid_activation_function(self, activation_input):
    # We are using sigmoid as an activation function here. We can replace it with any function (tanh,RELU etc)
    if activation_input.value > 500:
        return Node(1.0)
    elif activation_input.value < -500:
        return Node(0.0)

    neg_input = Node(-1.0) * activation_input
    exp_neg_input = neg_input.exp()  # Use the new exp() method
    one_plus_exp = Node(1.0) + exp_neg_input
    sigmoid_result = Node(1.0) / one_plus_exp
    return sigmoid_result

  def relu_activation_function(self, activation_input):
    if activation_input.value <= 0:
        out = Node(0.0)
        out._prev = {activation_input}

        def _backward():
            activation_input.gradient += 0.0 * out.gradient
        out._backward = _backward
        return out
    else:
        return activation_input

  def get_parameter(self):
    if self.input_layer_neuron:
        return []
    else:
      return self.weights + [self.bias]

  #This function calculates the neuron_value based on the previous_neurons and weights and biases and the activation function
  def __call__(self,input_value = 0):
     if self.input_layer_neuron:
      self.neuron_value = Node(input_value)
      return self.neuron_value

     weighted_sum = Node(0.0)
     for input_neuron,weight in zip(self.previous_neurons,self.weights):
      weighted_sum = weighted_sum + (input_neuron.neuron_value * weight)

     #Now we add some bias to the weighted sum
     activation_input = weighted_sum + self.bias

     #Now we pass the weighted sum to the activation function to calculate the final value of the neuron
     #The activation function makes sure that the value of the neuron stays between -1 and 1
     if self.output_layer:
      self.neuron_value = activation_input
     else:
      self.neuron_value = self.relu_activation_function(activation_input)
     return self.neuron_value
