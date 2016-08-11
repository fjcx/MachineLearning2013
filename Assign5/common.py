def dot(a, b):
  """Return the dot (inner) product of vectors a and b."""
  assert len(a) == len(b), "Lengths of %s and %s do not match" % (a, b)
  return sum((ai * bi) for ai, bi in zip(a,b))

def scale_and_add(v0, a, v1):
  """Scale v1 by a and add to v0."""
  assert len(v0) == len(v1)
  for i in range(len(v0)):
    v0[i] += a * v1[i]

def step(v):
  return 1 if v > 0 else 0

def average(lst):
  """Return average of an iterable of numbers."""
  total = 0.0
  count = 0
  for elt in lst:
    total += elt
    count += 1
  return total / count
