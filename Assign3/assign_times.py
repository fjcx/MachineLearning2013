# Student: Frank O'Connor
# E-80 Assignment 3
# Email: fjo.con@gmail.com

import csv, sys, os.path, bisect

assign_count = {}
availibility = {}
fail_reason = ""

def main(argv):

	# Example Usage: 'python .\assign_times.py shifts.txt'
	if len(sys.argv) < 2 or len(sys.argv) > 2:
		sys.exit('Usage: %s shifts_filename' % sys.argv[0])

	if not os.path.exists(sys.argv[1]):
		sys.exit('ERROR: Text file \"%s\" was not found!' % sys.argv[1])

	shifts_filename = sys.argv[1]

	ifile  = open(shifts_filename, "rb")
	
	# X set of variables, here they are hour shifts {0_1,0,_2,1_1,1_2,2_1,2_2,3_1,3_2}
	# note: 2 timeslots per shift
	# D set of domains, here they domains for the shifts will be students
	# C set of constraints, constraints are:
	#	2 students for every timeslot
	#	students must do 2 shifts
	
	#availibility = {}
	assignment = {}
	domains = {}
	
	
	for line in ifile:
		# remove trailing whitespace, \r, \n etc
		line = line.rstrip()
		# assuming can split lines on whitespace, as question states to assume 
		# "hour identifiers and names are opaque strings that contain no whitespace"
		split_words = line.split(" ")
		#print split_words
		student = split_words.pop(0)
		availibility[student] = split_words
		#print split_words
		for word in split_words:
			# we get all the timeslots, note every shift has 2 slots
			assignment[word+"_1"] = None
			assignment[word+"_2"] = None
			# construct variable domains also
			if word+"_1" in domains.keys():
				domains[word+"_1"].append(student)
				domains[word+"_2"].append(student)
			else:
				domains[word+"_1"] = [student]
				domains[word+"_2"] = [student]
			assign_count[student] = 0
			assign_count[student] = 0
				
	print assignment
	print availibility
	print "domains", domains
	# dict representing assignment of shifts, initially, none assigned
	
	# copy.copy(assignment) ??
	back_tracking_search(assignment, availibility, domains, assign_count)

	ifile.close()
	
def back_tracking_search(assignment, availibility, domains, assign_count):
	print_assignment(assignment)
	if is_complete_assignment(assignment, availibility):
		# complete valid assignment
		return assignment
	next_var = select_unassigned_variable(assignment, availibility, domains)
	ordered_domain_values = order_values(domains[next_var], assignment, assign_count)
	for dom_val in ordered_domain_values:
		print "in here", dom_val
		assignment[next_var] = dom_val
		assign_count[dom_val] = assign_count[dom_val] + 1
		if is_consistent(assignment, availibility):
			print "is consist -- in here"
			result = back_tracking_search(assignment, availibility, domains, assign_count)
			if result is not None:
				return result
		assignment[next_var] = None
		assign_count[dom_val] = assign_count[dom_val] -1
	return None	
	
def is_consistent(assignment, availibility):
	for timeslot, assigned_val in assignment.iteritems():
		if assigned_val is not None:
			# verify valid assigned matches availibility
			if timeslot[:-2] not in availibility[assigned_val]:
				print timeslot[:-2], assigned_val, " assignment not consistent"
				return False
		
		if timeslot.endswith("_1"):
			# same student can't be assigned to 2 timeslots in the same shift
			if assignment[timeslot[:-1]+"2"] is not None and assignment[timeslot] is not None:
				if assignment[timeslot[:-1]+"2"] == assignment[timeslot]:
					print "same shift"
					return False
	return True
	
def select_unassigned_variable(assignment, availibility, domains):
	minimum_remaining_values = None
	count = None
	# applying minimum remaining values heuristic
	for timeslot, possible_values in domains.iteritems():
		print "timeslot" , timeslot , "possible_values" , possible_values
		if assignment[timeslot] is None:
			if count is None:
				minimum_remaining_values = timeslot
				count = len(possible_values)
			elif len(possible_values) < count:
				count = len(possible_values)
				minimum_remaining_values = timeslot
			print count
	return minimum_remaining_values
	
def order_values(values, domains, assign_count):
	ordered_values = values
	# initial sort on number of domain values available, 
	#(this deals with tiebreaker on sort for on assigned values)
	#ordered_values = sorted(ordered_values, key=sort_by_availibility, reverse=True)
	## !!!!!!!!!! First search may not be necessary !!!!!!!!!!!
	print "ordered_values_availibility", ordered_values
	# order by least constraining value, 
	# i.e. the student with least assigned slots
	ordered_values = sorted(ordered_values, key=sort_by_assign_count)
	print "ordered_values", ordered_values
	return ordered_values
	
def sort_by_assign_count(s):
    print "assign_count", assign_count
    return assign_count[s]
	
def sort_by_availibility(s):
    print "len(availibility)", len(availibility[s])
    return len(availibility[s])
	
def is_complete_assignment(assignment, availibility):
	for timeslot, assigned_val in assignment.iteritems():
		if assigned_val is None:
			# All slots not assigned yet
			print "not all assigned"
			return False
		else:
			# verify valid assigned matches availibility
			if timeslot[:-2] not in availibility[assigned_val]:
				print timeslot[:-2], assigned_val, " assignment not valid"
				return False
		
		if timeslot.endswith("_1"):
			print timeslot
			# same student can't be assigned to 2 timeslots in the same shift
			if assignment[timeslot[:-1]+"2"] == assignment[timeslot]:
				print "same shift"
				return False
		# Also each student must have more than 2 ??
	print_assignment(assignment)
	return True
	
def print_assignment(assignment):
	
	for timeslot, assigned_val in sorted(assignment.iteritems()):
		if timeslot.endswith("_1"):
			firstPerson = assigned_val
			secondPerson = assignment[timeslot[:-1]+"2"]
			if firstPerson is None:
				firstPerson = "Nobody"
			if secondPerson is None:
				secondPerson = "Nobody"
			print "Hour "+ timeslot[:-2] + ": "+ firstPerson + " " + secondPerson
						
				
# --------- main ---------		
if __name__ == '__main__': 
    main(sys.argv)	
