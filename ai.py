from sklearn import tree
import argparse
import sys
import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonData", nargs='?')
    return parser


def calculateMaxTime(j):
	maxTime = 0
	for i in range(len(j["tasks"])):
		if maxTime < j["tasks"][str(i)]["control_time"]:
			maxTime = j["tasks"][str(i)]["control_time"]
	return maxTime


def findChoise(cool_team, cool_tasks_mass):
	minValue = 10
	minID = -1
	for i in range(len(cool_tasks_mass)):
		if abs(cool_tasks_mass[i] - cool_team) < minValue:
			minValue = abs(cool_tasks_mass[i] - cool_team) 
			minID = i

	return minID



parser = createParser()
namespace = parser.parse_args()

j = json.loads(namespace.jsonData)

with open('dataTeamsThousands.json') as data_file:    
    dataTeam = json.load(data_file)

XTeam = []
yTeam = []
for i in dataTeam['input']:
	x = []
	for k in i:
		x.append(k)
	XTeam.append(x)

for i in dataTeam['output']:
	yTeam.append(i)

clfTeam = tree.DecisionTreeClassifier()
clfTeam = clfTeam.fit(XTeam, yTeam)

time = j["team"]["time"]
maxTime = j["team"]["maxTime"]
numPeople = j["team"]["numPeople"]
maxNumPeople = j["team"]["maxNumPeople"]
complete = j["team"]["complete"]
allTasks = len(j["tasks"])
hintsUsed = j["team"]["hintsUsed"]
allHints = j["team"]["allHints"]

with open('dataTasksThousands.json') as data:    
    dataTasks = json.load(data)

XTasks = []
yTasks = []
for i in dataTasks['input']:
	x = []
	for k in i:
		x.append(k)
	XTasks.append(x)

for i in dataTasks['output']:
	yTasks.append(i)

clfTasks = tree.DecisionTreeClassifier()
clfTasks = clfTasks.fit(XTasks, yTasks)


cool_team = clfTeam.predict([1 - time/maxTime, 1 - numPeople/maxNumPeople, complete/allTasks, 1-hintsUsed/allHints])

maxCt = calculateMaxTime(j)

#print (a)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         sks_cool"]
cool_tasks_mass = []
for i in range(len(j["tasks"])):
	controlTime = j["tasks"][str(i)]["control_time"]
	taskType = j["tasks"][str(i)]["taskType"]
	numOfQuestions = j["tasks"][str(i)]["numOfQuestions"]
	subDifficult = j["tasks"][str(i)]["subDifficult"]
	cool_task = clfTasks.predict([controlTime / maxCt, taskType, numOfQuestions / 10, subDifficult / 10])
	cool_tasks_mass.append(cool_task[0])


choise = findChoise(cool_task, cool_tasks_mass)

print(j["tasks"][str(choise)]["id"])