#Create a function that takes an integer and returns the factorial of that integer.
#That is, the integer multiplied by all positive lower integers.
def factorial(num):
	if (num ==1):
		return 1;
	else:
		return num * factorial(num-1)
