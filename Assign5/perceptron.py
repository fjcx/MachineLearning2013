# Student: Frank O'Connor
# E-80 Assignment 3
# Email: fjo.con@gmail.com

import common, copy

class NotConverged(Exception):
  print 'Algorithm did not converge'

def learn(examples, max_iterations=100, raiseExcep=True):
  """Return a perceptron learned from [([feature], class)].

  Args:
    examples: a list of pairs of a list of features and a class variable.
      Features should be numbers, class should be 0 or 1.
    max_iterations: number of iterations to train.  Gives up afterwards

  Returns:
    A list of weights, one per feature, plus one more feature.  See predict
      for the interpretation of the weights.

  Raises:
    NotConverged, if training did not converge within the provided number
      of iterations.
  """
  
  # getting estimate of initial feature set size from passed data
  input_size = len(examples[0][0])
  # setting all initial weights to 0
  # and adding dummy weight for threshold value
  weights = [0.0 for _ in xrange(input_size+1)]
  converged = False
  iteration = 0
  while not converged:
   # keeping track of error count
   error_count = 0
   # for input feature set
   for input_x, desired_y in examples: 
    inputs = list(input_x)
	# adding dummy placeholder value as threshold
    inputs.append(1)

	# predicting the value with the current weights
	# getting dot product of weights and features, then applying step function
    predicted_p = common.step(common.dot(inputs, weights))
	
	# increase error count if we have a difference between our desired 
	#output and predicted output
    error = desired_y - predicted_p

    if error != 0:
     error_count += 1
	 # adjust inputs by adding the error * feature value to each respective weight
     for index, input_value in enumerate(inputs):
      weights[index] += error * input_value

	# printing adjusted weights
	# commented out for part 4 ( as too much text was displayed)
    #print 'weights:', weights

   iteration += 1
   #print '\n'
   # stopping criteria
   if iteration >= max_iterations or error_count == 0: 
    print '# of iterations:',iteration
    if iteration >= max_iterations and raiseExcep:
	 raise NotConverged('Perceptron did not converge')
    converged = True

  return weights

def predict(weights, features):
  """Return the prediction given perceptron weights on an example.

  Args:
    weights: A vector of weights, [w1, w2, ... wn, t], all numbers
    features: A vector of features, [f1, f2, ... fn], all numbers

  Returns:
    1 if w1 * f1 + w2 * f2 + ... * wn * fn + t > 0
    0 otherwise
  """
  fcpy = list(features)
  # adding dummy value for threshold
  fcpy.append(1)

  # getting dot product of features and weights
  dot = common.dot(fcpy, weights)
  # applying step function (output 1 or 0)
  predicted_p = common.step(dot)
  
  # commented out for part 4 ( as too much text was displayed)
  #print 'Input Weights' , weights
  #print 'Input Features' , features
  #print 'Predicted Result', predicted_p
  #print '\n'
  
  return predicted_p
