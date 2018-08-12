import argparse
import numpy as np
from matplotlib import pyplot as plt
from TrainMLP import InitMLP, LoadMLP, SaveMLP, DuplicateMLP, TrainMLP_Batch
from TestMLP import InferenceMLP

GAMMA = 0.9
REWARD_WIN = 1			# immediate reward when winning the game.
REWARD_LOSE = -1		# immediate reward when losing the game.
REWARD_TIE = 0.1		# immediate reward when the game is a tie.
REWARD_TWOWAYS = 0		# immediately extra reward when it is a two-way winning board for computer.

TRAIN = []
REPLAYSIZE = 2048		# Replay memory size: number of latest experiences
BATCHSIZE = 32			# Minibatch size
MORE_REPETITIONS = 1	# Number of repetitions for non-zero-reward states added to replay memory.

# indexes of  8-ways to win the tic tac toe game.
CHAMPIONS = ((0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6))

def InitBoard():
	return [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

# Check if any player wins the game. Return the wining piece, or '*' for tie, or ' ' for game not finished.
def CheckChampion(board):
	for (b0, b1, b2) in CHAMPIONS:
		if board[b0] != ' ' and board[b0] == board[b1] == board[b2]:
			return board[b0]
	else:
		return ' ' if ' ' in board else '*'

# Randomly find an empty square index.
def RandomMove(board):
	while True:
		n = np.random.randint(0, 9)
		if board[n] == ' ':
			return n

# Convert the board to MLP input vector.
def BoardToVector(board):
	boardbits = {'x': [0, 1], 'o': [1, 0], ' ': [0, 0]}
	vector = []
	for b in board:
		vector += boardbits[b]
	return vector

# Inference MLP to get the rewards for all actions.
def InferenceQ(mlp, board):
	vector = BoardToVector(board)
	qlist = InferenceMLP(mlp, vector)
	return qlist

# Define reward for the input board in the piece's point of view.
def Reward(board, piece):
	opponent = 'x' if piece == 'o' else 'o'
	winner = CheckChampion(board)
	if winner == piece:
		return REWARD_WIN
	elif winner == opponent:
		return REWARD_LOSE
	elif winner == '*':
		return REWARD_TIE
	else:
		return 0

# Check if input board is game over.
def IsTerminal(board):
	return True if CheckChampion(board) != ' ' else False

# Check if there no way for opponent and at least 2 ways for the piece to win the game.
def IsTwoWayWinning(board, piece):
	# If no extra reward for 2-way winning, return no 2-way winning immediately.
	if REWARD_TWOWAYS == 0:
		return False
	# Perform costly 2-way winning check.
	opponent = 'x' if piece == 'o' else 'o'
	nWin = 0
	for champ in CHAMPIONS:
		line = [board[b] for b in champ]
		if line.count(' ') == 1:
			if line.count(opponent) == 2:
				return False
			if line.count(piece) == 2:
				nWin += 1
	return True if nWin >=2 else False

# Determine actions for the input in epsilon-greedy manner.
def DetermineAction(mlp, board, epsilon):
	if np.random.uniform(0, 1) < epsilon:
		index = RandomMove(board)
	else:
		qlist = InferenceQ(mlp, board)
		index = FindBestMove(board, qlist)
	return index

# Find the index of the highest probabilities among all the empty squares.
def FindBestMove(board, qlist):
	bestIndex = start = board.index(' ')
	for i in range(start + 1, len(qlist)):
		if board[i] == ' ' and qlist[i] > qlist[bestIndex]:
			bestIndex = i
	return bestIndex

# Put the new example at head of the replay memopry, and delete the oldest ones at tail.
# More repetitions are put into replay memory for terminal state s'.
def AddExample(example):
	terminal = example[5]
	if not terminal:
		TRAIN.insert(0, example)
	else:
		for i in range(MORE_REPETITIONS):
			TRAIN.insert(0, example)
	# Remove oldest examples to limit the replay memory size.
	while len(TRAIN) > REPLAYSIZE:
		del TRAIN[-1]
	
# Find the max quality score of the next move in the given board.
def MaxQ(mlp, board):
	qlist = InferenceQ(mlp, board)
	index = FindBestMove(board, qlist)
	return qlist[index]

# Double-DQN's maxQ
def MaxQ_DDQN(mlp, mlp_minus, board):
	qlist = InferenceQ(mlp, board)
	index = FindBestMove(board, qlist)
	qlist = InferenceQ(mlp_minus, board)
	return qlist[index]	

# Update Q function by training MLP with a minibatch of data.
def UpdateQ(mlp, mlp_minus, alpha, useDDQN):
	# Return if data is not enough for reliably sampling a minibatch. 
	if len(TRAIN) < REPLAYSIZE // 4 or len(TRAIN) < BATCHSIZE:
		return 0	
	# Randomly pick a minibatch plus the newly added example at TRAIN[0] to train MLP. Note to find MaxQ using mlp_minus.
	train = TRAIN[1:]
	np.random.shuffle(train)
	train = train[:BATCHSIZE-1]
	train.insert(0, TRAIN[0])
	minibatch = []
	for (board, piece, index, reward, board_p, isTerminal) in train:
		qlist = InferenceQ(mlp, board)
		if isTerminal:
			maxQ = 0
		else:
			maxQ = MaxQ(mlp_minus, board_p) if not useDDQN else MaxQ_DDQN(mlp, mlp_minus, board_p)
		qlist[index] = reward + GAMMA * maxQ
		minibatch.append({'train': BoardToVector(board), 'target': qlist})
	error = TrainMLP_Batch(mlp, minibatch, alpha)
	return error

# Computer makes movement based on MLP inference.
def ComputerMove(mlp, board, piece, epsilon):
	index = DetermineAction(mlp, board, epsilon)
	board_p = list(board)
	board_p[index] = piece
	return index, board_p

# Trainer makes random movement.
def TrainerMove(board, piece):
	index = RandomMove(board)
	board_p = list(board)
	board_p[index] = piece
	return board_p

# Train computer's MLP in one game episode.
def TrainComputer(mlp, mlp_minus, alpha, epsilon, piece, firstPlayer, useDDQN):
	board = InitBoard()
	error, nTrainCount = 0, 0
	opponent = 'x' if piece == 'o' else 'o'
	if not firstPlayer:
		board = TrainerMove(board, opponent)
	while not IsTerminal(board):
		index, board_temp = ComputerMove(mlp, board, piece, epsilon)
		board_p = board_temp if IsTerminal(board_temp) else TrainerMove(board_temp, opponent)
		reward =  REWARD_TWOWAYS if IsTwoWayWinning(board_p, piece) else Reward(board_p, piece)
		AddExample((board, piece, index, reward, board_p, IsTerminal(board_p)))
		error += UpdateQ(mlp, mlp_minus, alpha, useDDQN)
		nTrainCount += 1
		board = board_p
	return board, error/nTrainCount

def main():
	global REPLAYSIZE, BATCHSIZE, REWARD_TWOWAYS, MORE_REPETITIONS, GAMMA
	# Parse command line arguments
	parser = argparse.ArgumentParser(description = 'Deep Q Learning Tic Tac Toe training tool.')
	parser.add_argument("TrainPlayer", help = "training the first player (1) or the second (2).")
	parser.add_argument("Episodes", help = "number of rounds to train the tic tac toe game.")
	parser.add_argument("OutModel", help = "output model.")
	parser.add_argument("-im", "--initmodel", help = "optional initial NN model.")
	parser.add_argument("-tp", "--topology", help = "number of neurons in each layer, example '18-36-9'")
	parser.add_argument("-dd", "--useDDQN", help = "use Double-DQN algorithm, default using DQN.", action = "store_true")
	parser.add_argument("-ep", "--epsilon", help = "start epsilon-greedy probability, default 0.999.")
	parser.add_argument("-ga", "--gamma", help = "Deep Q Learning discount factor, default %.2f."%GAMMA)
	parser.add_argument("-lr", "--learnrate", help = "learning rate alpha, default 0.1.")
	parser.add_argument("-lrdf", "--learnratedecayfactor", help = "learning rate decay factor, default 0.9.")
	parser.add_argument("-lrds", "--learnratedecaystep", help = "steps to decay the learning rate, default 50 steps.")
	parser.add_argument("-rp", "--replaysize", help = "replay memory size, default %d."%REPLAYSIZE)
	parser.add_argument("-mb", "--minibatch", help = "minibatch size, default %d."%BATCHSIZE)
	parser.add_argument("-rr", "--rewardrepetition", help = "repetition of none-zero-rewrd states in the replay memory, default %d."%MORE_REPETITIONS)
	parser.add_argument("-rt", "--rewardtwoways", help = "extra reward for 2-way winning, default %f."%REWARD_TWOWAYS)
	parser.add_argument("-ms", "--milestone", help = "number of milestones (save result) in the training process, default no milestone.")
	parser.add_argument("--noplot", help = "do not plot result chart", action = "store_true")
	parser.add_argument("-v", "--verbosely", help = "verbosely showing the training progress", action = "store_true")
	args = parser.parse_args()

	# Initialize MLP, tanh activation function and linear output.
	if args.initmodel:
		mlp = LoadMLP(args.initmodel)
	elif args.topology:		
		board = InitBoard()
		dim = len(BoardToVector(board))
		layers = [int(s) for s in args.topology.split('-')]
		assert layers[0] == dim and layers[-1] == len(board), "Invalid topology '%s': input and output dimensions have to be %d and %d respectively."%(args.topology, dim, len(board))
		mlp = InitMLP(layers, 'tanh', 'linear')
	else:
		raise Exception("Neither '--initmodel/-im' nor '--topology/-tp' is found.")

	# Determine computer's turn.
	if args.TrainPlayer == '1':
		piece, firstPlayer = 'x', True
	elif args.TrainPlayer == '2':
		piece, firstPlayer = 'o', False
	else:
		raise Exception('Invalid argument for TrainPlayer [%s], has to be 1 or 2.'%args.TrainPlayer)

	# Deterrmine replay memory size and minibatch size.
	if args.replaysize: 
		REPLAYSIZE = max(1, int(args.replaysize))
	if args.minibatch:
		BATCHSIZE = max(1, int(args.minibatch))
	assert BATCHSIZE <= REPLAYSIZE, 'Invalid arguments for "--replaysize/-rp" or "--minibatch/-mb".'

	# Determine two-way extra reward and repetitions of training data for states of non-zero rewards.
	if args.rewardtwoways:
		REWARD_TWOWAYS = float(args.rewardtwoways)
	if args.rewardrepetition:
		MORE_REPETITIONS = max(1, int(args.rewardrepetition))

	# Determine number of episodes and milestone size, after which MLP will be saved to file.
	nEpisodes = int(args.Episodes)
	nMiles = max(1,  nEpisodes // int(args.milestone)) if args.milestone and int(args.milestone) > 0 else nEpisodes

	# Determine MLP's learning rate alpha, and its decay factor and step size.
	ALPHA = alpha = max(0, float(args.learnrate)) if args.learnrate else 0.1
	lrDecayStep = max(1,  nEpisodes // int(args.learnratedecaystep)) if args.learnratedecaystep else max(1, nEpisodes // 50)
	lrDecayFactor = max(0.1, min(1, float(args.learnratedecayfactor))) if args.learnratedecayfactor else 0.9

	# Determine DQN discount factor gamma, ranging from 0 to 1.
	if args.gamma:
		GAMMA = max(0, min(1, float (args.gamma)))

	# Determine start epsilon greedy probability.
	epsilon = min(1, max(0, float(args.epsilon))) if args.epsilon else 0.999

	mlp_minus = DuplicateMLP(mlp)
	errors = []
	for ep in range(nEpisodes):
		board, err = TrainComputer(mlp, mlp_minus, alpha, epsilon, piece, firstPlayer, args.useDDQN)
		if err > 0:
			errors.append(err)
		# Annealing epsilon (the exploration probability) in 50 steps.
		if ep > 0 and ep % max(1, (nEpisodes//50)) == 0:
			epsilon *= 0.9
		# Annealing alpha (MLP learning rate) gradually.
		if ep > 0 and ep % lrDecayStep == 0:
			alpha *= lrDecayFactor
		# Verbosely output the training progress
		if args.verbosely:
			winner = CheckChampion(board)
			print(ep, ':', 'winner=%s error=%.6f'%(winner, err))
		# Save MLP temporarily after each milestone.
		if ep > 0 and ep % nMiles == 0:
			SaveMLP(mlp, '~ms~%d~%s'%(ep//nMiles, args.OutModel))
		# Update mlp_p used to estimate MaxQ(.) after a while to avoid over-estimation of MaxQ.
		mlp_minus = DuplicateMLP(mlp)

	# Save the finally trained MLP.
	SaveMLP(mlp, args.OutModel)

	# Discard the first 5% unstable errors before plotting.
	errors = errors[len(errors)//20:]

	if not args.noplot:
		title = 'Training %s error: %.4f\n'%(args.OutModel, errors[-1])
		title += 'topology: %s, '%('-'.join(str(n) for n in mlp['layers']))
		title += 'hidden: %s'%(mlp['actfunc'].__name__)
		if mlp['outfunc'] != mlp['actfunc']:
			title += ', out: %s'%(mlp['outfunc'].__name__)
		plt.title(title, fontsize = 12)
		xlabel = 'replay: %d, mb: %d, '%(REPLAYSIZE, BATCHSIZE)
		xlabel += 'alpha: %f(%d,%.2f), '%(ALPHA, lrDecayStep, lrDecayFactor)
		xlabel += 'gamma: %.2f'%GAMMA
		plt.xlabel('episodes (' + xlabel + ')', fontsize = 11)
		plt.ylabel('error', fontsize = 12)
		plt.plot(errors)
		plt.show()

if __name__ == '__main__':
	main()

