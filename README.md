# NanoGrad
I have always wondered how a neural network actually works.

Inspired by Andrej Karpathy's Micrograd, this is my attempt to build a simple neural network from scratch. 

# Problem Statement

A simple database query such as Select * from Database_Name Where column > 7 has some execution time associated with it. This execution time usually depends on the following fields:
  1. Number of parameters in the Where clause
  2. The number of rows scanned by the query

I want to create a NN which takes these fields as an input and predicts the execution time of any SQL query.


