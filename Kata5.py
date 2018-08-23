#Create a function that takes a list of numbers and returns the following statistics:

#   - Minimum Value
#   - Maximum Value
#   - Sequence Length
#   - Average Value

def minMaxLengthAverage(lst):
	lista = []
	lista.append(min(lst))
	lista.append(max(lst))
	lista.append(len(lst))
	lista.append(sum(lst)/len(lst))
	return lista

#Create a function that takes a list of numbers and returns the sum of the two lowest positive numbers. 

def sum_two_smallest_nums(lst):
	lista = sorted(lst)
	new = []
	for i in lista:
		if i>=0:
			new.append(i)
	return (new[0]+new[1])

#Create a function that takes a list of numbers and returns a list where each number is the sum of itself + all previous numbers in the list.

def cumulative_sum(lst):
	new=[]
	a=0;
	for i in lst:
		a+=i
		new.append(a)
	return new
