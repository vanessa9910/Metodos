#Create a function that takes a list of numbers and returns only the even values.

def noOdds(lst):
	final = []
	for i in range(len(lst)):
			if lst[i] % 2 == 0:
				final.append(lst[i])
	return final

#Create a function that takes a list of numbers and returns the mean value.
def mean(lst):
	final = 0
	for i in range (len(lst)):
		num= lst[i]
		final+=num
	return round(final/len(lst),2)

#Create a function that takes a string and returns the number (count) of vowels contained within it.
def count_vowels(txt):
	a = txt.count("a")
	e = txt.count("e")
	i = txt.count("i")
	o = txt.count("o")
	u = txt.count("u")
	return a+e+i+o+u
