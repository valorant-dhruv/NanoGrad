#This class represents a single node
#Note that each input,weight and a bias is a Node
class Node:
  
  #Constructor for this class
  def __init__(self,value):

    #Each node has some value associated with it
    if not isinstance(value, (int, float)):
            raise TypeError("Neuron value must be a number (int or float).")
    
    self.value = float(value)

    #Each node also has a value called gradient which is derivative of that node with respect to the output value
  #This is the function to print the node
  def __repr__(self):
    return f"Node(value={self.value})"
