#!/usr/bin/env python
"""
Train and predict using a Hidden Markov Model part-of-speech tagger.

Usage:
  hw6.py training_file test_file
"""

import argparse
import collections
import operator
import math

# Smoothing methods
NO_SMOOTHING = 'None'  # Return 0 for the probability of unseen events
ADD_ONE_SMOOTHING = 'AddOne'  # Add a count of 1 for every possible event.
# *** Feel free to add additional methods here ***
GOOD_TURING_SMOOTHING = 'GoodTuring'  #

# Unknown word handling methods
PREDICT_ZERO = 'None'  # Return 0 for the probability of unseen words
# If p is the most common part of speech in the training data,
# Pr(unknown word | p) = 1; Pr(unknown word | <anything else>) = 0
PREDICT_MOST_COMMON_PART_OF_SPEECH = 'MostCommonPos'
# *** Feel free to add additional methods here ***

HiddenMarkovModel = collections.namedtuple('HiddenMarkovModel', (
    # Order 0 -> unigram model, order 1 -> bigram, order 2 -> trigram, etc.
    'order',
    # Emission probabilities, a map from (pos, word) to Pr(word|pos)
    'emission',
    # Transition probabilities
    # For a bigram model, a map from (pos0, pos1) to Pr(pos1|pos0)
    'transition',
    # A list of parts of speech known by the model
    'parts_of_speech',
    # *** Feel free to add additional fields to this tuple to store other ***
    # *** necessary information ***
    ))


def train_baseline(training_data):
  '''Train a baseline most-common-part-of-speech classifier.

  Args:
    training_data: a list of pos, word pairs:

  Returns:
    dictionary, default, where dictionary is a map from a word to the most
      common part-of-speech for that word, and default is the most common
      overall part of speech.
  '''
  # *** IMPLEMENT ME ***

  count_data = collections.defaultdict(list)
  overall_pos_count =[]
  result_dictionary = collections.defaultdict(list)

  # count pos occurences for each word
  for pair in training_data:
    # converting to lower case (using case folding)
	#count_data[pair[1].lower()].append(pair[0])
	# appear to get better baseline with no case-folding !!!
    count_data[pair[1]].append(pair[0])
    overall_pos_count.append(pair[0])

  #print count_data
  for word in count_data:		# maybe can instead count in intial loop !!!
    #print word, collections.Counter(count_data[word]).most_common(1)[0]
    result_dictionary[word] = collections.Counter(count_data[word]).most_common(1)[0][0]
  
  most_common_pos = collections.Counter(overall_pos_count).most_common(1)[0][0]
  # may need to null check !!!

  # seems to work for unit_test
  return result_dictionary, most_common_pos


def train_hmm(training_data, smoothing, unknown_handling, order):
  '''Train a hidden-Markov-model part-of-speech tagger.

  Args:
    training_data: A list of pairs of a word and a part-of-speech.
    smoothing: The method to use for smoothing probabilities.
       Must be one of the _SMOOTHING constants above.
    unknown_handling: The method to use for handling unknown words.
       Must be one of the PREDICT_ constants above.
    order: The Markov order; the number of previous parts of speech to
      condition on in the transition probabilities.  A bigram model is order 1.

  Returns:
    A HiddenMarkovModel instance.
  '''
  # *** IMPLEMENT ME ***
  
  emission_probs = collections.defaultdict()
  transition_probs = collections.defaultdict()
  parts_of_speech = []
  # temp, need to refactor!!
  list_of_words = []
  
  count_data = collections.defaultdict(list)
  overall_pos_count =[]
  emiss_dictionary = collections.defaultdict(list)
  trans_dictionary = collections.defaultdict(list)
  prev_pos = None

  # count pos occurences for each word
  for pair in training_data:
	if pair[0] not in parts_of_speech:
	  parts_of_speech.append(pair[0])
	if pair[1] not in list_of_words:
	  list_of_words.append(pair[1])
	
	# count transition values
	if prev_pos is not None:
	  if not trans_dictionary[prev_pos]:
	    trans_dictionary[prev_pos] = [{pair[0]: 1}, 1]
	  else:
	    if pair[0] in trans_dictionary[prev_pos][0]:
	      trans_dictionary[prev_pos][0][pair[0]] +=1
	    else:
	      trans_dictionary[prev_pos][0][pair[0]] = 1
	    trans_dictionary[prev_pos][1] += 1
	  
	prev_pos = pair[0]
	
	# count emissions values  
	if not emiss_dictionary[pair[0]]:
	  emiss_dictionary[pair[0]] = [{pair[1]: 1}, 1]
	else:
	  if pair[1] in emiss_dictionary[pair[0]][0]:
	    emiss_dictionary[pair[0]][0][pair[1]] +=1
	  else:
	    emiss_dictionary[pair[0]][0][pair[1]] = 1
	  emiss_dictionary[pair[0]][1] += 1

  # AT THIS POINT MAY want to ADD ONE to each total and add instance of pos->pos and (word,pos) for unknowns

  print 'smoothing', smoothing
  #print 'emiss_dictionary', emiss_dictionary
  #print 'trans_dictionary', trans_dictionary
  #smoothing = ADD_ONE_SMOOTHING
  # Add smoothing - start
  if smoothing == ADD_ONE_SMOOTHING:
    for pos in emiss_dictionary:
      for word in list_of_words:
        if word not in emiss_dictionary[pos][0]:
          emiss_dictionary[pos][0][word] = 1
        else:
		  emiss_dictionary[pos][0][word] += 1
        emiss_dictionary[pos][1] += 1

    for word in trans_dictionary:
      for pos in parts_of_speech:
        if pos not in trans_dictionary[word][0]:
          trans_dictionary[word][0][pos] = 1
        else:
		  trans_dictionary[word][0][pos] += 1
        trans_dictionary[word][1] += 1

  # NO_SMOOTHING -- NEEDS Review !!!!!	--> No Smoothing should prob just pick another option e.g. N ???
  # using good turing to smooth lower occurences (threshold at frequency of 6)
  if smoothing == GOOD_TURING_SMOOTHING:	
    for pos in emiss_dictionary:
	  one_occurence_count=0
	  #print 'emiss_dictionary[pos][0]',emiss_dictionary[pos][0]
	  #d=sorted(emiss_dictionary[pos][0].items(), key=lambda x:x[1])
	  #print 'emiss_dictionary[pos][0]',d
	  # Z {1: 7, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 10: 1, 11: 1})
	  # Z {0: 7, -- N1/N
	  #1: 7, -- (c+1) N2/N1 = 2*(1/7) = 2/7
	  #2: 1, -- = 3*(1/1) = 3
	  #3: 1, 4: 1, 5: 1, 6: 1, 10: 1, 11: 1})
	  freq_count = collections.defaultdict(list)
	  for word, occurence_count in emiss_dictionary[pos][0].iteritems():
	    if not freq_count[occurence_count]:
	      freq_count[occurence_count] = 1
	    else:
	      freq_count[occurence_count] += 1
	  print 'Bef--freq_count',pos, freq_count
	  
	  if not freq_count[1]:
	    freq_count[0] = 0
	  else:
	    freq_count[0] = freq_count[1]
	    next_index = 2
	    prev_index = 1
	    while prev_index < 6:
		  if freq_count[prev_index]:
		    print 'freq_count[prev_index]', freq_count[prev_index]
		    while not freq_count[next_index] and next_index<7:
		      next_index+=1
		    if freq_count[next_index] and freq_count[prev_index]:
		      freq_count[prev_index] = (prev_index+1)*(float(freq_count[next_index])/freq_count[prev_index])
		  prev_index += 1
		
	    print 'Aft--freq_count', freq_count
	  #print '1occWord',pos,one_occurence_count, float(one_occurence_count)/emiss_dictionary[pos][1]	# allocation prob for unknown word for the pos
  # Add smoothing - end
  #print 'trans_dictionary', trans_dictionary
  
  for pos in emiss_dictionary:
    for word in emiss_dictionary[pos][0]:
	  emission_probs[(pos,word)] = math.log(float(emiss_dictionary[pos][0][word])/float(emiss_dictionary[pos][1]))
	  
  for prev_pos in trans_dictionary:
    for pos in trans_dictionary[prev_pos][0]:
	  transition_probs[(prev_pos,pos)] = math.log(float(trans_dictionary[prev_pos][0][pos])/trans_dictionary[prev_pos][1])

  return HiddenMarkovModel(order, emission_probs, transition_probs, parts_of_speech)


def find_best_path(lattice):
  """Return the best path backwards through a complete Viterbi lattice.

  Args:
    FOR ORDER 1 MARKOV MODELS (bigram):
      lattice: [{pos: (score, prev_pos)}].  See compute_lattice for details.

  Returns:
    A list of parts of speech.  Does not include the <s> tokens surrounding
    the sentence, so the length of the return value is 2 less than the length
    of the lattice.
  """
  # *** IMPLEMENT ME ***
  result_path = []
  prev_var = '<s>'
  
  # !!!! perhaps more logic is needed here ???? PICKING MAX or something ???!!!!
  for layer_ind in  xrange(len(lattice)-1,1,-1):
    layer_choice = lattice[layer_ind][prev_var][1]
    prev_var = layer_choice
    result_path.insert(0, layer_choice)
	
  return result_path


def compute_lattice(sentence, model):
  """Compute the Viterbi lattice for an example sentence.

  Args:
    sentence: a list of words, not including the <s> tokens on either end.
    model: A HiddenMarkovModel instance.

  Returns:
    FOR ORDER 1 Markov models:
    lattice: [{pos: (score, prev_pos)}]
      That is, lattice[i][pos] = (score, prev_pos) where score is the
      log probability of the most likely pos/word sequence ending in word i
      having part-of-speech pos, and prev_pos is the part-of-speech of word i-1
      in that sequence.

      i=0 is the <s> token before the sentence
      i=1 is the first word of the sentence.
      len(lattice) = len(sentence) + 2.

    FOR ORDER 2 Markov models: ??? (extra credit)

  """
  # *** IMPLEMENT ME ***
  # Confirm sentence actually has values in it
  if sentence:
    sentence.append('<s>')
  lattice = [{'<s>': (math.log(1), None)}]

  for sentence_index, word in enumerate(sentence):
    lattice_layer = dict()
    for pos in model.parts_of_speech:
	  if (pos, word) in model.emission:	# !!! this is all possible values for the word !!!??!!
	    emiss_prob = model.emission[(pos, word)]
	    trans_probs = dict()
	    for prev_state_key, value in lattice[sentence_index].iteritems():
	      # check if prev state can actually transition to current state
	      if (prev_state_key,pos) in model.transition:
	        trans_probs[prev_state_key]=model.transition[(prev_state_key,pos)] + emiss_prob + lattice[sentence_index][prev_state_key][0]
	      else:
	        # if transition prob does not exist then set it as close to zero prob of transition
			# this should only occur when have no smoothing !!!
	        trans_probs[prev_state_key]=float("-inf") # using in place of log prob of zero
		
		if not trans_probs:
		  print lattice[sentence_index], model.transition
	    prev_max_trans_key = max(trans_probs, key=trans_probs.get)
	    lattice_layer[pos] = (trans_probs[prev_max_trans_key], prev_max_trans_key)
	  #else:
	    # word does not have chance of being this pos, according to training data

    if not lattice_layer:	# !!!! case where word has no EMISSION PROB for any pos (i.e. Unknown word)!!!
      # saying most common is only option, so prob of 1
	  emiss_prob = math.log(1)
	  trans_probs = dict()
	  # !!!!! USING 'N' AS most COMMON, BUT MAY WANT to find actual most common !!!!
	  for prev_state_key, value in lattice[sentence_index].iteritems():
	    # check if prev state can actually transition to current state
	    if (prev_state_key,'N') in model.transition:
	      trans_probs[prev_state_key]=model.transition[(prev_state_key,'N')] + emiss_prob + lattice[sentence_index][prev_state_key][0]
	    else:
		  # if transition prob does not exist then set it as close to zero prob of transition
	      trans_probs[prev_state_key]=float("-inf")	# using in place of log prob of zero
		
	  prev_max_trans_key = max(trans_probs, key=trans_probs.get)
	  lattice_layer['N'] = (trans_probs[prev_max_trans_key], prev_max_trans_key)
	
	# add layer to lattice
    lattice.append(lattice_layer)
  
  return lattice


def read_part_of_speech_file(filename):
  '''Read a part-of-speech file and return a list of (pos, word) pairs.'''
  with open(filename) as pos_file:
    return [line.split() for line in pos_file]


def get_predictions(test_filename, predict_sentence):
  '''Given an HMM, compute predictions for each word in the test data.'''
  sentence = []
  true_poses = []
  for true_pos, word in read_part_of_speech_file(test_filename):
    if word != '<s>':
      sentence.append(word)
      true_poses.append(true_pos)
    else:
      predictions = predict_sentence(sentence)
      for word, pos, true_pos in zip(sentence, predictions, true_poses):
        yield pos, word, true_pos
      yield ('<s>', '<s>', '<s>')
      sentence = []
      true_poses = []


def compute_score(predictions):
  '''Compute the score for a set of predictions.

  Args:
    predictions: a list of predicted-part-of-speech, word, true-pos triples.

  Returns:
    number of correct predictions, total number of predictions
    Does not count sentence-end tokens.
  '''
  num_correct = 0
  num_words = 0
  for pos, _, true_pos in predictions:
    if pos == '<s>':
      continue
    num_words += 1
    if true_pos == pos:
      num_correct += 1
  return num_correct, num_words


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('train_filename')
  parser.add_argument('test_filename')
  parser.add_argument('--smoothing', choices=(NO_SMOOTHING, ADD_ONE_SMOOTHING, GOOD_TURING_SMOOTHING),
      default=NO_SMOOTHING)
  parser.add_argument('--order', default=1, type=int)
  parser.add_argument('--unknown',
      choices=(PREDICT_ZERO, PREDICT_MOST_COMMON_PART_OF_SPEECH,),
      default=PREDICT_ZERO)
  parser.add_argument('--print_score', action='store_true')
  args = parser.parse_args()
  training_data = read_part_of_speech_file(args.train_filename)
  if args.order == 0:
    dictionary, default = train_baseline(training_data)
    def predict_sentence(sentence):
      return [dictionary.get(word, default) for word in sentence]
  else:
    model = train_hmm(read_part_of_speech_file(args.train_filename),
        args.smoothing, args.unknown, args.order)
    def predict_sentence(sentence):
      return find_best_path(compute_lattice(sentence, model))
  predictions = get_predictions(args.test_filename, predict_sentence)
  if args.print_score:
    num_correct, num_words = compute_score(predictions)
    print 'Accuracy: %d/%d = %.2f%%' % (num_correct, num_words,
        100.0 * num_correct / num_words)
  else:
    for prediction, word, true_pos in predictions:
      print prediction, word, true_pos

if __name__ == '__main__':
  main()
