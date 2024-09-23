Pet project which serves as a playground. Predict solubility of organic molecules using GNN

The project isn't finished yet and mostly serves as a playground for me to get familiar with deep neural networks.

The current implementation consists of:

Database from https://moleculenet.org/datasets-1 Created a dataset using PyTorch Geometric Each data point is represented by a graph with node(atom) and edge(bonds) features
Graph processing is based on the paper "Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification" (https://arxiv.org/abs/2009.03509) Each layer consists of this Transformer Convolutional layer followed by batch normalisation. The number of these layers is one of the hyperparameters and should be determined.
After that graph representation is reduced using the mean pooling. Then 2 fully connected linear layers are present with RELU non-linear function.
This gives a single scalar which is used in the mean squared error function
Adam optimization algorithm is used

This implementation is very raw and requires much work and effort. Right now I am playing around with different architectures for the model to find the best one. Then I plan to perform a hyperparameter search for the selected model. Then I will possibly create some lightweight server for it, a convenient docker container etc.
