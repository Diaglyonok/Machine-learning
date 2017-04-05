import json
from pprint import pprint
from sklearn import tree
import random


with open('task_cool.json') as data_file:    
    data = json.load(data_file)

X = []
y = []
for i in data['input']:
	x = []
	for k in i:
		x.append(k)
	X.append(x)

for i in data['output']:
	y.append(i)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

def randfloat():
	return "{0:.2f}".format(round(random.randint(0, 100) / 100., 2))

def randfloatmi():
	return "{0:.2f}".format(1 - round(random.randint(0, 100) / 100., 2))

def randType():
	a = random.randint(1, 3)
	return a*0.33

def rand_num_quest(type):
	if type == 0.33:
		return "{0:.1f}".format(round(random.randint(1, 10)/10., 2))
	else:
		return 0.1

Xout = []
yout = []

for i in range(0, 10000):
	type = randType()
	first_arg = float(randfloat()); sec_arg = float(type); third_arg = float(rand_num_quest(type)); fourth_arg = float(randfloat())
	Xout.append([first_arg, sec_arg, third_arg, fourth_arg])
	a = clf.predict([first_arg, sec_arg, third_arg, fourth_arg])
	yout.append(int(a[0]))

"""def randfloat():
	return "{0:.2f}".format(round(random.randint(0, 100) / 100., 2))

def randfloatmi():
	return "{0:.2f}".format(1 - round(random.randint(0, 100) / 100., 2))

Xout = []
yout = []

for i in range(0, 10000):
	first_arg = float(randfloatmi()); sec_arg = float(randfloatmi()); third_arg = float(randfloat()); fourth_arg = float(randfloatmi())
	Xout.append([first_arg, sec_arg, third_arg, fourth_arg])
	a = clf.predict([first_arg, sec_arg, third_arg, fourth_arg])
	print(a[0])
	yout.append(int(a[0]))
"""

print (Xout)
print (yout)

with open('dataTasksThousands.json', 'w') as outfile:
    json.dump({'input' : Xout, 'output': yout}, outfile)

