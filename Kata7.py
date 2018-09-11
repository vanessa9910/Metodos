#Create a function that takes an integer and returns the factorial of that integer.
#That is, the integer multiplied by all positive lower integers.
def factorial(num):
	if (num ==1):
		return 1;
	else:
		return num * factorial(num-1)

#Take a list of integers (positive or negative or both) and return the sum of the absolute value of each element.

def get_abs_sum(lst):
	suma = 0;
	for i in range (len(lst)):
		suma+=abs(lst[i]);
	return suma
