# NanoGrad

I've always wondered how a neural network actually works under the hood. Inspired by Andrej Karpathy's micrograd, this project is my attempt to build a simple neural network framework from scratch, focusing on the fundamental principles.

# Problem Statement

Consider a simple database query, such as SELECT * FROM Database_Name WHERE column > 7. This query has an associated execution time. This execution time typically depends on a few key factors:

  1. The number of parameters in the WHERE clause.
  2. The number of rows scanned by the query.

My goal is to create a neural network that takes these two fields as input and predicts the execution time of any given SQL query.
