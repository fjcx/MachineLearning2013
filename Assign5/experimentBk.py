#!/usr/bin/env python
'''
Perform simple train/test machine learning experiment on ARFF-formatted data.

Usage:
  experiment.py data.arff neural_network|perceptron [-t 0.9]
'''

import argparse
import random
import re
import sys

import neural_network
import perceptron

class BadLine(Exception):
  '''An exception representing an error parsing an ARFF file.'''


class NominalAttribute(object):
  '''Represent an ARFF nominal attribute'''
  def __init__(self, name, values):
    self.name = name
    self.values = values
    self.hidden_observed = False

  def check_value(self, value):
    '''Check an attribute value in an ARFF file for validity.'''
    if value == '?':
      self.hidden_observed = True
    elif value not in self.values:
      raise BadLine('Unsupported value %s for feature %s' % (value, self.name))

  def binarize(self, value):
    '''Convert an arff value to a list of binary values (see binarize below).'''
    assert value in self.values + ['?',], (
        'Unsupported value %s for feature %s' % (value, self.name))
    if len(self.values) == 2 and not self.hidden_observed:
      return [
          0 if value == self.values[0] else
          1 if value == self.values[1] else
          None]
    else:
      return [1 if value == allowed_value else 0
          for allowed_value in self.values]

def binarize(data, attributes, class_attribute):
  '''Convert a sparse dataset to binary (0/1) features.

  The input is nominal feature values, e.g. red/green/blue.  The output
  are all binary.  A feature with more than two values is converted to
  three binary presence features (e.g. red -> [1,0,0], green -> [0,1,0]).
  A missing value is converted to all-0 features ('?' -> [0,0,0]).

  A two-valued feature will be converted to one binary feature
  (red/green; red -> 0, green -> 1) unless there are missing values, in which
  case it is converted just as a feature with more than two values.

  Args:
    data: a list of (features, output) pairs.
    attributes: a list of attribute objects.
    class_attribute: an attribute object.

  Returns:
    a list of (features, outputs) pairs, where all features and outputs
       are 0/1.
  '''
  ret = []
  for features, output in data:
    binary_features = []
    for value, attribute in zip(features, attributes):
      binary_features += attribute.binarize(value)
    ret.append((binary_features, class_attribute.binarize(output)))
  return ret


class Perceptron(object):
  '''A class representing a learned, multi-output perceptron.'''

  def __init__(self, data):
    num_outputs = len(data[0][1])
    self.models = [
        perceptron.learn([(features, outputs[i]) for features, outputs in data])
        for i in range(num_outputs)]

  def predict(self, features):
    '''Return predictions for each output.'''
    return [perceptron.predict(model, features) for model in self.models]


class NeuralNetwork(object):
  '''A class representing a learned neural network.'''
  def __init__(self, data):
    self.model = neural_network.learn(data)

  def predict(self, features):
    '''Return 0/1 predictions based on neural network model.'''
    return [1 if p > 0.5 else 0
        for p in neural_network.predict(self.model, features)]


ALGORITHMS = {
    'neural_network': NeuralNetwork,
    'perceptron': Perceptron,
    }


READING_ATTRIBUTES = 0
READING_DATA = 1

def read_arff(filename):
  '''Read an ARFF machine learning file, return dataset
  Args:
    filename: A string path to an .arff file
  Returns:
    [[feature], [output]], [attribute], class_attribute -
        a list of pairs of features and outputs, a list of attribute objects,
        and a class attribute object.
  '''
  data = []
  line_number = 0
  with open(filename) as arff:
    attributes = []
    state = READING_ATTRIBUTES
    for line in arff:
      line_number += 1
      try:
        if line[0] == '%':
          continue  # Skip comments.
        elif line.strip() == '':
          continue  # Skip blanks.
        elif line.upper().startswith('@RELATION'):
          if state == READING_DATA:
            raise BadLine('Relation in data section')
          # This line defines the name of the relation.  We don't care.
        elif line.upper().startswith('@ATTRIBUTE'):
          if state != READING_ATTRIBUTES:
            raise BadLine('Attribute in data')
          match = re.search(r'@ATTRIBUTE\s+(\S+)\s+\{([^\}]+)\}', line, re.I)
          if match:
            name, values = match.groups()
            values = [value.strip() for value in values.split(',')]
            attributes.append(NominalAttribute(name, values))
          else:
            raise BadLine('Unsupported attribute type')
        elif line.upper().startswith('@DATA'):
          state = READING_DATA
        else:
          if state != READING_DATA:
            raise BadLine('data section begun before @DATA')
          values = [feature.strip() for feature in line.split(',')]
          if len(values) != len(attributes):
            raise BadLine('number of features != number of attributes')
          for value, attribute in zip(values, attributes):
            attribute.check_value(value)
          # Final feature is class feature
          data.append((values[:-1], values[-1]))
      except BadLine as exception:
        print 'Error reading input file %s on line %d %r: %s' % (
            filename, line_number, line.strip(), exception)
        sys.exit(1)
  return data, attributes[:-1], attributes[-1]


def random_split(data, train_fraction):
  '''Randomly split a dataset into training and test data.

  Args:
    data: a list of examples
    train_fraction: a number from 0 to 1

  Returns:
    train, test - two disjoint sublists of the data representing a partition
        into train_fraction and (1-train_fraction) of the data, randomly
        permuted.
  '''
  assert 0 <= train_fraction <= 1
  data_with_key = [(random.random(), datum) for datum in data]
  data_with_key.sort()
  randomized_data = [datum for _, datum in data_with_key]
  split_point = int(train_fraction * len(data))
  return randomized_data[:split_point], randomized_data[split_point:]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_filename')
  parser.add_argument('learning_algorithm', choices=ALGORITHMS.keys())
  parser.add_argument('-t', '--train_fraction', default=0.9)
  parser.add_argument('-p', '--predict_on_training', default=False,
      action='store_true')
  parser.add_argument('-d', '--print_data', default=False,
      action='store_true')
  args = parser.parse_args()
  algorithm = ALGORITHMS[args.learning_algorithm]
  data, attributes, class_attribute = read_arff(args.data_filename)
  print 'data', data
  data = binarize(data, attributes, class_attribute)
  if args.print_data:
    print data
  if args.predict_on_training:
    train = test = data
  else:
    train, test = random_split(data, args.train_fraction)
  model = algorithm(train)
  num_correct = 0
  for features, outputs in test:
    prediction = model.predict(features)
    num_correct += 1 if prediction == outputs else 0
  print 'Accuracy: %d / %d = %.2f%%' % (
      num_correct, len(test), 100.0 * num_correct / len(test))


if __name__ == '__main__':
  main()
