import numpy as np
from collections import defaultdict

def reward(fname):
	reward_list = np.zeros((81, 1), dtype = int)
	i = 0
	for l in open(fname):
		reward_list[i][0] = int(l)
		i += 1
	return reward_list

def load_matrix(fname):
	action = np.zeros((81, 81), dtype = float)
	for l in open(fname):
		l = l.split()
		action[int(l[0]) - 1, int(l[1]) - 1] = float(l[2])
	return action
				
def get_max(action, state, v_k):
	max_value = -10000.00
	max_action = -1
	for i in range(4):
		temp = 0.0
		for k in range(81):
			temp += action[i][state][k] * v_k[k]
		if temp > max_value:
			max_value = temp
			max_action = i
	return max_value, max_action

def get_identity_matrix(n):
	identity = np.zeros((n, n), dtype = int)
	for i in range(n):
		identity[i][i] = 1 
	return identity

def get_matrix_policy(action, pi_):
	transition_matrix = np.zeros((81, 81), dtype = float)
	#s = i, pi(s) = pi_[i], 
	for i in range(81):
		a = pi_[i]
		transition_matrix[i] = action[a][i]
	return transition_matrix

def policy_optimization(action, v, rewards):
	pi_ = [0] * 81
	identity_matrix = get_identity_matrix(81)
	for k in range(30):
		
		transition_matrix = get_matrix_policy(action, pi_)
		old_v = np.matrix(identity_matrix - 0.9925 * transition_matrix).I * rewards
		for i in range(81):
			#print i, get_max(action, i, old_v)
			max_value, max_action = get_max(action, i, old_v)
			v[i] = rewards[i][0] + 0.9925 * max_value
			pi_[i] = max_action
	return v, pi_

def value_optimization(action, v, rewards):
	pi_ = [0] * 81
	identity_matrix = get_identity_matrix(81)
	for k in range(30):
		old_v = list(v)
		
		for i in range(81):
			#print i, get_max(action, i, old_v)
			max_value, max_action = get_max(action, i, old_v)
			v[i] = rewards[i][0] + 0.9925 * max_value
			pi_[i] = max_action
	return v, pi_



rewards = reward('rewards.txt')
action1 = load_matrix('prob_a1.txt')	
action2 = load_matrix('prob_a2.txt')
action3 = load_matrix('prob_a3.txt')
action4 = load_matrix('prob_a4.txt')
action = [action1, action2, action3, action4]

v = [0.0] * 81
value, pi_policy_iteration = policy_optimization(action, v,  rewards)
print value, "value"
value, pi_value_iteration = value_optimization(action, v, rewards)
print pi_value_iteration, "pi_value_iteration", value, "Value"

