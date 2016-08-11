# Student: Frank O'Connor
# E-80 Assignment 1
# Email: fjo.con@gmail.com

import csv, sys, os.path, bisect

def main(argv):

	# Example Usage: 'python .\find_routes.py tstflights.csv BOS UCS'
	if len(sys.argv) < 4:
		sys.exit('Usage: %s flights-database origin-airport algorithm' % sys.argv[0])

	if not os.path.exists(sys.argv[1]):
		sys.exit('ERROR: Database %s was not found!' % sys.argv[1])

	flights_database = sys.argv[1]
	origin_airport = sys.argv[2]
	algorithm = sys.argv[3]
	
	# set algorithm we are using
	if algorithm == 'BFS':
		algo = breadth_first_search
		print 'Breadth First Search:'
	elif algorithm == 'DFS':
		algo = depth_first_search
		print 'Depth First Search:'
	elif algorithm == 'UCS':
		algo = uniform_cost_search
		print 'Uniform Cost Search:'
	else:
		sys.exit("ERROR: Not a valid algorithm name. Please choose from algorithms 'BFS', 'DFS', 'UCS'")
	
	ifile  = open(flights_database, "rb")
	reader = csv.reader(ifile)
	
	flight_graph = Graph()
	# create set of all possible destinations, only searching for destinations that are possible to reach
	destination_set = set()
	for row in reader:
		destination_set.add(row[1])
		# create graph of all connections from csv file
		flight_graph.connect1(row[0], row[1], int(row[2]))

	ifile.close()
	
	# store total stats for calculating averages
	total_time = 0.0
	total_memory = 0.0
	total_length = 0.0
	result_count = 0.0
		
	# loop through and search for route to each destination in destination_set from origin
	for dest_airport in destination_set:		
	
		#prob = InstrumentedProblem(GraphProblem(origin_airport, dest_airport, flight_graph))
		prob = GraphProblem(origin_airport, dest_airport, flight_graph)
		result = algo(prob)
		# it is my assumuption the 'time' is interpreted as number of goal_tests (i.e nodes 'explored')
		if result is not None:
			result_count += 1
			total_time += prob.goal_tests
			total_memory += prob.max_mem
			total_length += result.path_cost			
			# printing time, memory, length, list for airports between origin and dest
			print prob.goal_tests, prob.max_mem, result.path_cost, origin_airport, ' '.join(result.solution())
		else:
			print 'No route for' , origin_airport , '->', dest_airport
	
	if result_count	!= 0.0:
		print 'Average: %.2f %.2f %.2f' % (total_time/result_count, total_memory/result_count, total_length/result_count)
	print 'distinct dest results: %d' % result_count
		
					
# --------- GraphProblem class (modified from aimi search.py) ---------
# modifications to class to be more specific to assignment implementation
class GraphProblem(object):
    "The problem of searching a graph from one node to another."
    def __init__(self, initial, goal, graph):
		self.initial = initial
		self.goal = goal
		self.graph = graph
		self.goal_tests = 0
		self.memory = 0
		self.max_mem = 1

    def actions(self, A):
        "The actions at a graph node are just its neighbors."
        return self.graph.get(A).keys()
		
    def result(self, state, action):
        "The result of going to a neighbor is just that neighbor."
        return action
		
    def goal_test(self, state):
		self.goal_tests += 1
		return state == self.goal

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A,B) or infinity)
		
    def h(self, node):
        "h function is straight-line distance from a node's state to goal."
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return infinity
									 
# --------- breadth_first_search (modified from aimi search.py) ---------
# modifications to calculate memory cost
def breadth_first_search(problem):
    node = Node(problem.initial,None,None)
    problem.memory = 1	# add initial Node to memory count
    if problem.goal_test(node.state):
        return node
    frontier = FIFOQueue()
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        explored.add(node.state)
        child_list = node.expand(problem)
        problem.memory += len(child_list) # add expanded list of Nodes to memory count
		# if current memory usage is larger set, then record it
        if problem.memory > problem.max_mem:
			problem.max_mem = problem.memory
        for child in child_list:
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
					return child
                frontier.append(child)
    return None
	
# --------- depth_first_search (modified from aimi search.py) ---------		
# modifications to calculate memory cost
def depth_first_search(problem):
    "Search the deepest nodes in the search tree first."
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    If two paths reach a state, only use the first one."""
    frontier = Stack()
    frontier.append(Node(problem.initial))
    problem.memory = 1	# add initial Node to memory count
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        child_list = node.expand(problem)
        if len(child_list) is not 0:
	        problem.memory += len(child_list) # if node has children, add expanded list of Nodes to memory count
        else:
			if len(frontier) is not 0:
				par = node.parent
				problem.memory -= 1	# remove leaf node from mem count
				# walk back along path and decrement mem count for nodes/path no longer kept in memory
				while par is not problem.initial and par is not frontier[-1].parent:
					par = par.parent
					problem.memory -= 1 # remove parent from mem count if no longer used
		# if current memory usage is larger set, then record it
        if problem.memory > problem.max_mem:
			problem.max_mem = problem.memory
        frontier.extend(child for child in child_list
                        if child.state not in explored
                        and child not in frontier)
    return None

# --------- uniform_cost_search (modified from aimi search.py) ---------	
# modifications to calculate memory cost	
def uniform_cost_search(problem):
    return best_first_graph_search(problem, lambda node: node.path_cost)
	
def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    problem.memory = 1	# add initial Node to memory count
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        child_list = node.expand(problem)
        problem.memory += len(child_list) # if node has children, add expanded list of Nodes to memory count
		# if current memory usage is larger set, then record it
        if problem.memory > problem.max_mem:
			problem.max_mem = problem.memory
        for child in child_list:
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None
	
# --------- GraphProblem Class from aimi search.py (not modified) ---------	
class Graph:
    """A graph connects nodes (verticies) by edges (links).  Each edge can also
    have a length associated with it.  The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C.  You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added.  You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B.  'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, dict=None, directed=True):
        self.dict = dict or {}
        self.directed = directed
        if not directed: self.make_undirected()

    def connect1(self, A, B, distance):
        "Add a link from A to B of given distance, in one direction only."
        self.dict.setdefault(A,{})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.dict.setdefault(a, {})
        if b is None: return links
        else: return links.get(b)

    def nodes(self):
        "Return a list of nodes in the graph."
        return self.dict.keys()
		
# --------- Node Class from aimi search.py (not modified) ---------
class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        update(self, state=state, parent=parent, action=action,
               path_cost=path_cost, depth=0)
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state, action, next))

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

# --------- from aimi utils.py (not modified) ---------
infinity = 1.0e400

def update(x, **entries):
    """Update a dict; or an object with slots; according to entries.
    >>> update({'a': 1}, a=10, b=20)
    {'a': 10, 'b': 20}
    >>> update(Struct(a=1), a=10, b=20)
    Struct(a=10, b=20)
    """
    if isinstance(x, dict):
        x.update(entries)
    else:
        x.__dict__.update(entries)
    return x
	
def memoize(fn, slot=None):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, store results in a dictionary."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if not memoized_fn.cache.has_key(args):
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]
        memoized_fn.cache = {}
    return memoized_fn

def some(predicate, seq):
    """If some element x of seq satisfies predicate(x), return predicate(x).
    >>> some(callable, [min, 3])
    1
    >>> some(callable, [2, 3])
    0
    """
    for x in seq:
        px = predicate(x)
        if px: return px
    return False
	
class Queue:
    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(order, f): Queue in sorted order (default min-first).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self):
        abstract

    def extend(self, items):
        for item in items: self.append(item)
	
class FIFOQueue(Queue):
    """A First-In-First-Out Queue."""
    def __init__(self):
        self.A = []; self.start = 0
    def append(self, item):
        self.A.append(item)
    def __len__(self):
        return len(self.A) - self.start
    def extend(self, items):
        self.A.extend(items)
    def pop(self):
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A)/2:
            self.A = self.A[self.start:]
            self.start = 0
        return e
    def __contains__(self, item):
        return item in self.A[self.start:]
		
def Stack():
    """Return an empty list, suitable as a Last-In-First-Out Queue."""
    return []
	
class PriorityQueue(Queue):
    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x).
    Also supports dict-like lookup."""
    def __init__(self, order=min, f=lambda x: x):
        update(self, A=[], order=order, f=f)
    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))
    def __len__(self):
        return len(self.A)
    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]
    def __contains__(self, item):
        return some(lambda (_, x): x == item, self.A)
    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item
    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)
                return
				
# --------- main ---------		
if __name__ == '__main__': 
    main(sys.argv)	
