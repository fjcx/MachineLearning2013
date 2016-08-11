"""Unit tests for the perceptron module."""
import unittest

import perceptron
import common_test

class Test(unittest.TestCase):
  def test_predict(self):
    self.assertEqual(perceptron.predict([1, 2, 0], [0, 0]), 0)
    self.assertEqual(perceptron.predict([1, 2, 0], [1, 1]), 1)
    self.assertEqual(perceptron.predict([1, 2, 0], [1, -1]), 0)
    self.assertEqual(perceptron.predict([1, 2, 0], [-1, 1]), 1)
    self.assertEqual(perceptron.predict([1, 2, -4], [1, 1]), 0)
    self.assertEqual(perceptron.predict([1, 2, -3], [1, 1]), 0)
    self.assertEqual(perceptron.predict([1, 2, -2], [1, 1]), 1)

  def test_learn_xor(self):
    self.assertRaises(perceptron.NotConverged, perceptron.learn,
        common_test.XOR)

  def test_learn_or(self):
    data = common_test.OR
    model = perceptron.learn(data)
    for x, y in data:
      self.assertEqual(perceptron.predict(model, x), y)

  def test_learn_and(self):
    data = common_test.AND
    model = perceptron.learn(data)
    for x, y in data:
      self.assertEqual(perceptron.predict(model, x), y)

  def test_learn_if(self):
    data = common_test.IF
    model = perceptron.learn(data)
    for x, y in data:
      self.assertEqual(perceptron.predict(model, x), y)
