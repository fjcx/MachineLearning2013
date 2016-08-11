"""Unit tests for the neural_network module."""
import unittest

import neural_network
import common_test

def make_multiple_outputs(data):
  return [(features, [output]) for features, output in data]

def clip(v):
  """Convert a neural network output to a 0/1 prediction."""
  return 0 if v < 0.5 else 1

class Test(unittest.TestCase):
  longMessage=True

  _SIMPLE_MODEL = neural_network.NeuralNetwork(
        activation=neural_network.sigma,
        weights=[
          [
          # weights into hidden unit 0
            [0.3, 0.4],
          # weights into hidden unit 1
            [-0.5, -0.6],
          ],
          # weights into output unit
          [[0.1, 0.2]]])
  _COMPLEX_MODEL = neural_network.NeuralNetwork(
        activation=neural_network.sigma,
        weights=[
          [
            # weights into hidden unit 0
            [0.3, 0.4],
            # weights into hidden unit 1
            [-0.5, -0.6],
            # weights into hidden unit 2
            [0.9, -1.0],
          ],
          [
            # weights into output unit 0
            [0.1, 0.2, 1.1],
            # weights into output unit 1
            [0.7, -0.8, -1.2],
          ]])
  def test_get_unit_values_simple(self):
    inputs = [0, 0]
    unit_values = neural_network.get_unit_values(self._SIMPLE_MODEL, inputs)
    self.assertEqual(len(unit_values), 3)  # three layers of units
    self.assertEqual(unit_values[0], inputs)  # layer 0 is inputs
    self.assertEqual(len(unit_values[1]), 2)  # layer 1 is hidden units
    self.assertAlmostEqual(unit_values[1][0], 0.5)
    self.assertAlmostEqual(unit_values[1][1], 0.5)
    self.assertEqual(len(unit_values[2]), 1)  # layer 2 is output units
    self.assertAlmostEqual(unit_values[2][0], 0.5374, places=4)

  def test_get_unit_values_complex(self):
    inputs = [-1.4, 1.3]
    unit_values = neural_network.get_unit_values(self._COMPLEX_MODEL, inputs)
    self.assertEqual(len(unit_values), 3)  # three layers of units
    self.assertEqual(unit_values[0], inputs)  # layer 0 is inputs
    self.assertEqual(len(unit_values[1]), 3)  # layer 1 is hidden units
    self.assertAlmostEqual(unit_values[1][0], 0.5250, places=4)
    self.assertAlmostEqual(unit_values[1][1], 0.4800, places=4)
    self.assertAlmostEqual(unit_values[1][2], 0.0718, places=4)
    self.assertEqual(len(unit_values[2]), 2)  # layer 2 is output units
    self.assertAlmostEqual(unit_values[2][0], 0.5566, places=4)
    self.assertAlmostEqual(unit_values[2][1], 0.4744, places=4)

  def test_calculate_errors_simple(self):
    unit_values = neural_network.get_unit_values(self._SIMPLE_MODEL, [0, 0])
    errors = neural_network.calculate_errors(self._SIMPLE_MODEL, unit_values, 
        [0])
    self.assertEqual(len(errors), 2)  # Hidden errors, output errors
    self.assertEqual(len(errors[0]), 2)  # One error per hidden unit
    self.assertAlmostEqual(errors[0][0], -0.0033, places=4)
    self.assertAlmostEqual(errors[0][1], -0.0067, places=4)
    self.assertEqual(len(errors[1]), 1)  # One error per output unit
    self.assertAlmostEqual(errors[1][0], -0.1336, places=4)

  def test_learn_or(self):
    data = make_multiple_outputs(common_test.OR)
    model = neural_network.learn(data)
    for x, (y,) in data:
      self.assertEqual(clip(neural_network.predict(model, x)[0]), y,
          msg='datum %s, %s' % (x, y))

  def test_learn_and(self):
    data = make_multiple_outputs(common_test.AND)
    model = neural_network.learn(data)
    for x, (y,) in data:
      self.assertEqual(clip(neural_network.predict(model, x)[0]), y,
          msg='datum %s, %s' % (x, y))

  def test_learn_xor(self):
    data = make_multiple_outputs(common_test.XOR)
    model = neural_network.learn(data)
    for x, (y,) in data:
      self.assertEqual(clip(neural_network.predict(model, x)[0]), y,
          msg='datum %s, %s' % (x, y))
