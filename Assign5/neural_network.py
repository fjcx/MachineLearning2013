# Student: Frank O'Connor
# E-80 Assignment 3
# Email: fjo.con@gmail.com

"""Learning and prediction functions for artificial neural networks."""

import collections
import common
import math
import random
import sys

# Throughout this file, layer 0 of a neural network is the inputs, layer -1 is the outputs.

# weights is a list of lists of lists of numbers, where
# weights[a][b][c] is the weight into unit b of layer a+1 from unit c in layer a
NeuralNetwork = collections.namedtuple('NeuralNetwork', ('activation', 'weights'))

def get_unit_values(model, features):
  """Calculate the activation of each unit in a neural network.

  Args:
    model: a NeuralNetwork object
    features: a vector of feature values

  Returns:
    units, a list of lists of numbers, where
      units[a][b] is the activation of unit b in layer a
  """
  # init units
  units = [list(features)]
  layer_count = 0
  
  for layer in model.weights:
   layer_activations = []
   # for the nodes in layer calculate the dot product of the weights and the inputs to the node, and apply activation function
   for unit_weights in layer:
    layer_activations.append(model.activation(common.dot(units[layer_count],unit_weights)))

   units.append(layer_activations)
   layer_count+=1
  
  return units


def predict(model, features):
  """Calculate the prediction of a neural network on one example

  Args:
    model: a NeuralNetwork object
    features: a vector of feature values

  Returns:
    A list of numbers, the predictions for each output of the network
        for the given example.
  """
  #print 'model', model
  #print 'features', features

  inputs = list(features)
  inputs.append(1)
  unit_values = get_unit_values(model, inputs)
  pred_out = unit_values[-1]
  #print 'p_out', pred_out
  
  return pred_out


def calculate_errors(model, unit_values, outputs):
  """Calculate the backpropagated errors for an input to a neural network.

  Args:
    model: a NeuralNetwork object
    unit_values: unit activations, a list of lists of numbers, where
      unit_values[a][b] is the activation of unit b in layer a
    outputs: a list of correct output values (numbers)

  Returns:
    A list of lists of numbers, the errors for each hidden or output unit.
        errors[a][b] is the error for unit b in layer a+1.
  """
  # init errors
  errors = []
  num_outputs = len(outputs)
  #print '\ndesired_outputs', outputs
  
  errors_out_layer = [0.0 for _ in xrange(num_outputs)]
  # iterating backwards through layers of units (ignoring output, and input layers)
  # back-propagating errors through network
  errors_hid_layer = []
  for j in xrange(num_outputs):
   p_out = unit_values[-1][j]
   # error = predicted_out(1 - predicted_out)(y - predicted_out)
   errors_out_layer[j] = p_out*(1-p_out)*(outputs[j]-p_out)
   
   for i in xrange(len(unit_values)-2, 0, -1):
    errors_hid_layer = []
    unit_count = 0
    for p_hid_unit in unit_values[i]:
     weight_hid_to_out = model.weights[i][j][unit_count]

     err_hid_unit = p_hid_unit*(1-p_hid_unit)*(errors_out_layer[j])*(weight_hid_to_out)
     errors_hid_layer.append(err_hid_unit)
     unit_count+=1

  errors.append(errors_hid_layer)
  errors.append(errors_out_layer)

  return errors

def sigma(v):
  return 1 / (1 + math.exp(-v))


def learn(data,
    num_hidden=16,
    initial_weight_function=random.random,
    activation=sigma,
    max_iterations=1000,
    learning_rate=1):
  """Learn a neural network from data.

  Args:
    data: a list of pairs of input and output vectors, both lists of numbers.
    num_hidden: the number of hidden units to use.
    initial_weight_function: a function to produce a random initial weight.
    activation: the activation function to use for each network node.
    max_iterations: the max number of iterations to train before stopping.
    learning_rate: a scaling factor to apply to each weight update.

  Returns:
    A NeuralNetwork object.
  """
  #Initialize weights to random small number
  #while not converged
  #for each input (x,y)
    #compute current prediction p_u of each unit u
    #err_out = p_out(1-p_out)(y-p_out)
    #for each hid h: err_h = p_h(1-p_h) w_h->out err_out
    #for each unit u: weight*->u += err_u input_u
	
  num_input_features = len(data[0][0])+1
  num_outputs = len(data[0][1])
	
  # weights from input layer to hidden layer, initialised to small random values
  hidden_layer_weights = [[initial_weight_function() for _ in xrange(num_input_features)] for _ in xrange(num_hidden)]
	
  # weights from hidden layer to output layer, initialised to small random values
  output_layer_weights = [[initial_weight_function() for _ in xrange(num_hidden)] for _ in xrange(num_outputs)]
  
  # initialise weights with small random values
  weights = [hidden_layer_weights, output_layer_weights]

  converged = False
  iteration = 0
  while not converged:
   error_count = 0
   # for input feature set
   for input_x, desired_y in data:
    inputs = list(input_x)
    inputs.append(1)

    neur_network = NeuralNetwork(activation, weights)
    
    # get unit values for all nodes in network	
    unit_values = get_unit_values(neur_network, inputs)
	
	# calculate error for nodes in network verus expected values
    errors = calculate_errors(neur_network, unit_values, desired_y)

    #print 'inputs', inputs    
    #print 'desired_y', desired_y
    #print 'err_out', errors[-1]

	# if error for output nodes is not 0, adjust weights
    if errors[-1] != 0:
     error_count += 1
     for layer_index, layer_weights in enumerate(weights):
	  for unit_index, unit_weights in enumerate(layer_weights):
	   for u_weight_index, u_weight in enumerate(unit_weights):
        #adjust weights by (learning rate * the unit value at the node * the back-propagted error at the node)
	    weights[layer_index][unit_index][u_weight_index] += learning_rate * unit_values[layer_index][u_weight_index] * errors[layer_index][unit_index]

   iteration += 1
   #print 'error_count', error_count
   #print '\n'
   # Stopping criteria
   if iteration >= max_iterations or error_count == 0:
    print 'iterations',iteration
    converged = True
	
  return neur_network
