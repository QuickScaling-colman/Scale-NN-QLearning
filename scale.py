import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output
import random

class State:
	""" resoConsumed - resource consumed (nvm, cpu, mem, dsk, net)
	    resoConfigured - resource configured (nvm, cpu, mem, dsk, net)
	    actualLatency - current latency 
	    expectedLatency - expected latency by SLO """

	def __init__(self, resoConsumed, resoConfigured , actualLatency, expectedLatency):
		self.resoConsumed = resoConsumed
		self.resoConfigured = resoConfigured
		self.actualLatency = actualLatency
		self.expectedLatency = expectedLatency

def initStatetable():
	""" The state is a 3-dimensional numpy array (10x5x2). You can think of the first two dimensions as number of vms (1-10)
	    and utilization ranges: 0) 0% - 20%
				    1) 20% - 40%
				    2) 40% - 60%	
				    3) 60% - 80%
				    4) 80% - 100%
	    The 3rd dimension encodes the presense application at that range (nvm/utilization).  Since there are 2 different possible
	    states (application in range or not), the 3rd dimension of the state contains vectors of length 2.
	state = np.zeros((10,5)
	return state

def scaleAPI(state, action):
	return state

def currentStateAPI():
	return State( np.array([1, random.random(), random.random(), random.random(), random.random()]), 
		      np.array([10, 1, 1, 1, 1]), 2.3333, 2.444)

def getCurrentState():
	return State( np.array([ 2, 0.20, 0.10, 0.23, 0.67 ]), np.array([ 5, 0.40, 0.71, 0.30, 0.922 ]), 2.3333, 2.444)

def makeAction(state, action):
	new_state = state.deepcopy(state)

	# scale up
	if action == 1:
		# maximum number of vms is 10
		if state.resoConsumed < 10:
			#new_state.resoConsumed[0]++
			new_state = scaleAPI(state, action)
	# scale down
	elif action == 2:
		# minimum number of vms is 1
		if state.resoConsumed > 1:
			#new_state.resoConsumed[0]--
			new_state = scaleAPI(state, action)

def getReward(state):
	yNorm = state.actualLatency / float(state.expectedLatency)
	scoreSLO = np.sign( 1 - yNorm ) * np.exp( np.absolute( 1 - yNorm ) )
	vResource = np.divide(state.resoConsumed, state.resoConfigured)
	uConstrained = np.max(vResource)
	scoreU = np.exp( 1 - uConstrained )
	reward = scoreSLO * scoreU
	return reward

def heuristicPolicy(state, y):
	topSLObound = (1 - y) * float(expectedLatency)
	lowSLOblound = y * float(expectedLatency)
	
	if lowSLOblound < state.actualLatency < topSLObound:
		action = 0 # no change
	elif state.actualLatency > topSLObound:
		action = 1 # scale up
	else:
		action = 2 # scale down
	
	return action 
	

""" An input layer of 100 units [because our state has a total of 100 elements, number of vms (1 -10) * 10 utiliztion ranges]
    2 hidden layers of 164 and 150 units
    output layer of 3, one for each of our possible actions (no-change, scale-up, scale-down)
"""
model = Sequential()
model.add(Dense(164, init='lecun_uniform', input_shape=(100,)))
model.add(Activation('relu'))

model.add(Dense(150, init='lecun_uniform'))
model.add(Activation('relu'))

model.add(Dense(3, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)


""" MAIN LOOP """

gamma = 0.9 #since it may take several moves to goal, making gamma high
epsilon = 1

state = currentStateAPI()
print state.resoConsumed
print state.resoConfigured

status = 1

#Let's run our Q function on S to get Q values for all possible actions
qval = model.predict(state.reshape(1,100), batch_size=1)

if (random.random() < epsilon): 
	# use heuristics
	action = heuristicPolicy(state)
	print ("Performing heuristic action")
else:
	#choose best action from Q(s,a) values
	action = (np.argmax(qval))
	print ("Performing best action from Q(s,a) values")

"""
while(status == 1):

	#We are in state S
	#Let's run our Q function on S to get Q values for all possible actions
	qval = model.predict(state.reshape(1,100), batch_size=1)

	if (random.random() < epsilon): 
		# use heuristics
		action = heuristicPolicy(state)
		print ("Performing heuristic action")
	else: 
		#choose best action from Q(s,a) values
		action = (np.argmax(qval))
		print ("Performing best action from Q(s,a) values")

	#Take action, observe new state S'
	new_state = makeAction(state, action)

	#Observe reward
	reward = getReward(new_state)

	#Get max_Q(S',a)
	newQ = model.predict(new_state.reshape(1,100), batch_size=1)
	maxQ = np.max(newQ)
	y = np.zeros((1,3))
	y[:] = qval[:]

	if reward == -1: #non-terminal state
		update = (reward + (gamma * maxQ))
	else: #terminal state
		update = reward

	y[0][action] = update #target output
	model.fit(state.reshape(1,100), y, batch_size=1, nb_epoch=1, verbose=1)
	state = new_state

	if reward != -1:
		status = 0

	if epsilon > 0.1 and epsilon < 0.7:
		epsilon -= (1/200)
"""
