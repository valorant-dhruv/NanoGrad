# NanoGrad

I've always wondered how a neural network actually works under the hood. Inspired by Andrej Karpathy's micrograd, this project is my attempt to build a simple neural network framework from scratch, focusing on the fundamental principles.

# Problem Statement

Consider a simple database query, such as SELECT * FROM Database_Name WHERE column > 7. This query has an associated execution time. This execution time typically depends on a few key factors:

  1. The number of parameters in the WHERE clause.
  2. The number of rows scanned by the query.

My goal is to create a neural network that takes these two fields as input and predicts the execution time of any given SQL query.

# Project Steps

## First step:

We need to create a Node class which contains the following fields:
  1. The value of the Node
  2. The value of the derivative of the output node with respect to this node

## Second Step

We need to create a neuron class which contains the following fields:
  1. The value of the neuron after it has been passed to the activation function
  2. Previous neurons this neuron is attached to
  3. The weights this neuron is given as an input
  4. Bias associated with this neuron
