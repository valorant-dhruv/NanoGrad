#This class represents a single node
#Note that each input,weight and a bias is a Node
class Node:

  def __init__(self,value):
     #Each node has some value associated with it
    if not isinstance(value, (int, float)):
            raise TypeError("Node value must be a number (int or float).")
    self.value = float(value)
     #Each node also has a value called gradient which is derivative of the output node with respect to this node
     #We are especially interested in finding the gradient of the weight Nodes
    self.gradient = 0.0
     # Add these for backpropagation
    self._backward = lambda: None
    self._prev = set()
    self._op = ''

  def exp(self):
    out = Node(math.exp(self.value))
    out._prev = {self}
    out._op = 'exp'

    def _backward():
        self.gradient += math.exp(self.value) * out.gradient
    out._backward = _backward

    return out

  #We add a function to find the addition of two numbers
  def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value + other.value)
        out._prev = {self, other}
        out._op = '+'

        def _backward():
            self.gradient += out.gradient
            other.gradient += out.gradient
        out._backward = _backward

        return out

  #To handle the case of other + self
  def __radd__(self, other):
        return self + other

  #We add a function to find the subtraction of two numbers
  def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value - other.value)
        out._prev = {self, other}
        out._op = '-'

        def _backward():
            self.gradient += out.gradient
            other.gradient += -out.gradient
        out._backward = _backward

        return out

  #To handle the case of other - self
  def __rsub__(self, other):
        return other + (-self)

  #We add a function to find the power of two numbers
  def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Only supporting int/float powers for now.")
        out = Node(self.value**other)
        out._prev = {self}
        out._op = f'**{other}'

        def _backward():
            self.gradient += other * (self.value**(other-1)) * out.gradient
        out._backward = _backward

        return out

  #We add a function to find the multiplication of two numbers
  def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.value * other.value)
        out._prev = {self, other}
        out._op = '*'

        def _backward():
            self.gradient += other.value * out.gradient
            other.gradient += self.value * out.gradient
        out._backward = _backward

        return out

  #To handle the case of other * self
  def __rmul__(self, other):
        return self * other

  def __truediv__(self, other): # self / other
        return self * other**-1

  def __rtruediv__(self, other): # other / self
        return other * self**-1

  #Add negative operation
  def __neg__(self):
        return self * -1

  #Add backward method for backpropagation
  #It uses topological sort
  def backward(self):
        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Go one variable at a time and apply the chain rule
        self.gradient = 1.0
        for v in reversed(topo):
            v._backward()

  #This is the function to print the node
  def __repr__(self):
    return f"Node(value = {self.value}, gradient = {self.gradient})"
