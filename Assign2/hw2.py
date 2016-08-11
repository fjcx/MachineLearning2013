# CSCI E-80 Fall 2013 Artificial Intelligence
# Assignment 2
# Student: Frank O'Connor
# Email: fjo.con@gmail.com 
#
# Modify this file however you wish; you must implement ChooseAction and
# ApplyAction, but you can add other functions, variables, or classes as you see
# fit.
#
# Throughout this file, the numbers -1, 0, and 1 designate
#    1: player "A"
# -1: player "B"
#    0: an empty board space
import copy

# Dict to keep track of searched utilities and transposed/symmetric boards utilities
trans_map = {}

def ChooseAction(board, player):
    """Return the first empty board space.
 
    Args:
        board: An array of 16 numbers, each representing a board position
        player: A number representing the next player to move

    Returns:
        A number 0 to 15 indicating the chosen move.
    """
	# list of all possible initial actions
    possible_actions = list(i for i in range(16) if board[i] == 0)
	
    # default action
    chosen_move = min(possible_actions)

    # Dict to keep track of searched utilities and transposed/symmetric boards utilities
    trans_map = {}
	
    # function to use minimax to chose next move, we pass the current board, the possible actions and the player
    chosen_move = min_max_dec(possible_actions, copy.copy(board), player)
    
    return chosen_move
 

# 'best_utility' is minimax function altered to allow for extra move by
# either player A or B. The function tries to maximise for A and 
# minimise for B. The function also, stores calulated utilities and their
# board's transpositions utilities in the trans_map dictionary.
def best_utility(board, player, alpha, beta):
	if is_terminal(board):
		return compute_utility(board)
	else:
		# get all possible actions at that state (expand node)
		possible_actions = list(i for i in range(16) if board[i] == 0)
		
		#store best utility (max for A, min for B)
		best_u = None
				
		for a in possible_actions:
		    next_state, next_player = ApplyAction(a, copy.copy(board), player)
			
		    if str(next_state) not in trans_map:
				u = best_utility(next_state, next_player, alpha, beta)
				trans_map[str(next_state)] = u
				# get the transpositions/symmetries of the current board, and 
				# record the current calculated utility for these too.
				reverse, symet, rev_symet = transpose(next_state)
				trans_map[str(reverse)] = u
				trans_map[str(symet)] = u
				trans_map[str(rev_symet)] = u
		    else:
				# use recorded utility if we have one
				u = trans_map[str(next_state)]
			
			# Apply Alpha-Beta pruning also
		    # player 'A' wants to maximise utility (best is max)
		    if player == 1:
				if best_u is None or u > best_u:
					best_u = u
				if beta is not None and u >= beta:
					return u
				if alpha is None or u > alpha:
					alpha = u
		    else: # player 'B' wants to minimise utility (best is min)
				if best_u is None or u < best_u:
					best_u = u
				if alpha is not None and u <= alpha:
					return u
				if beta is None or u < beta:
					beta = u
		return best_u

def min_max_dec(possible_actions, board, player):
	"""Given a state in a game, calculate the best move by searching
        forward all the way to the terminal states. """
	max_move = None
	max_u = None
	alpha = None
	beta = None
	
	trans_map = {}
	
	for a in possible_actions:
		next_state, next_player = ApplyAction(a, copy.copy(board), player)
		
		if str(next_state) not in trans_map:
			u = best_utility(next_state, next_player, alpha, beta)
			trans_map[str(next_state)] = u
			# get the transpositions/symmetries of the current board, and 
			# record the current calculated utility for these too.
			reverse, symet, rev_symet = transpose(next_state)
			trans_map[str(reverse)] = u
			trans_map[str(symet)] = u
			trans_map[str(rev_symet)] = u
		else:
			# use recorded utility if we have one
			u = trans_map[str(next_state)]

		# Apply Alpha-Beta pruning also
		if player == 1:
			# leaving this print out here, so if it's slow we can see it's doing something
			print 'A - thinking ...'
			if max_u is None or u > max_u:
				max_move = a
				max_u = u
			if beta is not None and u >= beta:
				return a
			if alpha is None or u > alpha:
				alpha = u
		else:
			print 'B - thinking ...'
			if max_u is None or u < max_u:
				max_move = a
				max_u = u
			if alpha is not None and u <= alpha:
				return a
			if beta is None or u < beta:
				beta = u		
				
	return max_move
	
def is_terminal(board):
	# check is this an end state
	if 0 not in board:
		return True
	return False
	
def compute_utility(board):
	return sum(board)

# get the transpositions/symmetries of the current board
def transpose(curr_board):
	reverse = curr_board[::-1]
	
	symet = [reverse[i] for i in range(12,16)]
	symet.extend([reverse[i] for i in range(8,12)])
	symet.extend([reverse[i] for i in range(4,8)])
	symet.extend([reverse[i] for i in range(0,4)])
	
	rev_symet = [curr_board[i] for i in range(12,16)]
	rev_symet.extend([curr_board[i] for i in range(8,12)])
	rev_symet.extend([curr_board[i] for i in range(4,8)])
	rev_symet.extend([curr_board[i] for i in range(0,4)])

	return reverse, symet, rev_symet
	
def ApplyAction(action, board, player):
    """Put one token in the appropriate space.

    Args:
        action: A number 0 to 15 indicating the chosen move (where a marble
                ought to be placed).
        board: An array of 16 numbers, each representing a board position
        player: A number representing the next player to move

    Returns:
        A pair of a new board (having placed the marble appropriately) and
                the next player to move.
    """
    
    # default next player
    next_player = -player

    # ensure move is an int not a string
    action = int(action)
    board[action] = player
    
    # determine if adjacent squares to action are surrounded
    check_pos = action+1
    if is_surrounded(check_pos, board, player):
		# Checking the right
		board[check_pos] = player
		next_player = player

    check_pos = action-1
    if is_surrounded(check_pos, board, player):
		# Checking the left
		board[check_pos] = player
		next_player = player
		
    check_pos = action+4 
    if is_surrounded(check_pos, board, player):
		# Checking the bottom'
		board[check_pos] = player
		next_player = player
		
    check_pos = action-4
    if is_surrounded(check_pos, board, player):
		# Checking the top
		board[check_pos] = player
		next_player = player

    return board, next_player
    

def is_surrounded(position, board, player):
	is_surrounded = False
	left_edges = [0, 4, 8, 12]
	right_edges = [3, 7, 11, 15]

	# check if position is valid and not already assigned a player
	if position > -1 and position < 16 and board[position] == 0:
		if position + 4 > 15 or board[position + 4] == player:
			if position - 4 < 0 or board[position - 4] == player:
				if position - 1 < 0 or position in left_edges or board[position - 1] == player:
					if position + 1 > 15 or position in right_edges or board[position + 1] == player:
						is_surrounded = True
					
	return is_surrounded
