# TrainMLP.py
# Author: Taihsuan Ho, 2018/06/07
# A simple multilayer perceptron training program.

import argparse
import sys
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Training data loaded in LoadTrainingData(). Each element in TRAIN[] is a training example, which is a dict 
# of key 'train' and 'target', both values are numpy arrays with one additional dimension of value 1 for bias.
TRAIN = []	

# ADAM optimization object, initialized in InitADAM() called by TrainMLP_Batch().
ADAM = {}	

def DSigmoid(v, *_):
	return v * (1 - v)

def DTanh(v, *_):
	return 1 - v * v

def DReLU(v, mlp):
	return (v > 0) * 1

def Sigmoid(v, *_):
	return 1 / (1 + np.exp(-v))

def Tanh(v, *_):
	return 2 / (1 + np.exp(-2*v)) - 1

def ReLU(v, mlp):
	return np.maximum(v, 0)

def Linear(v, *_):
	return v

def Softmax(v, *_):
	ev = np.exp(v - np.max(v))
	return ev / ev.sum()

def CalcError(ref, hyp):
	err = ref - hyp
	return (err * err).sum()

def InitADAM(layers):
	adam = {'beta1': 0.9, 'beta2': 0.999, 'beta1t': 0.9, 'beta2t': 0.999, 'epsilon': pow(10,-8),
			'mt':  CreateWeights(layers, randomize = False),	
			'vt':  CreateWeights(layers, randomize = False)}
	ADAM.update(adam)

# Init MLP object. If argument 'mlp' is specified, ignore argument 'layer' and keep the weights unchanged in the input mlp.
def InitMLP(layers, actfunc, outfunc, mlp = None):
	ACTFUNCS = {'tanh':(Tanh, DTanh), 'sigmoid':(Sigmoid, DSigmoid), 'relu':(ReLU, DReLU)}
	OUTFUNCS = {'linear': Linear, 'softmax': Softmax}
	if not mlp:
		mlp = {'layers': layers, 'weights': CreateWeights(layers)}
	# Assign activation and output functions.
	if actfunc.lower() not in ACTFUNCS:
		actfunc = 'sigmoid'
	if outfunc.lower() not in OUTFUNCS:
		outfunc = actfunc
	mlp['actfunc'], mlp['derivative'] = ACTFUNCS[actfunc.lower()] 
	mlp['outfunc'] = OUTFUNCS[outfunc.lower()] if outfunc.lower() in OUTFUNCS else mlp['actfunc']
	mlp['actfuncname'], mlp['outfuncname'] = actfunc.lower(), outfunc.lower()
	return mlp

# Create weights containing (n-1) matrixes for n-layer MLP. 
# Argument layers is a list of number of each layer, including the input layer. For example, [3, 4, 4, 2] means 
# 3 input neurons, 2 hidden layers each of 4 neurons, and 2 neurons in the output layer. Between two layer, there's
# an n-rows-by-m-columns matrix, where n is number of neurons at the subsequent layer and m for the current layer.
# matrix[i][j] is weight on the synapsis from the j-th neuron of the current layer to the i-th neuron of the subsequent layer.
def CreateWeights(layers, randomize = True):
	nLayer = len(layers)
	weights = []
	for i in range(nLayer - 1):
		# +1 for both row and column for the bias neuron (always output 1).
		nRow, nColumn = layers[i+1] + 1, layers[i] + 1
		matrix = np.random.rand(nRow, nColumn) if randomize else np.zeros(shape = (nRow, nColumn))
		weights.append(matrix)
	return weights

# Feed forward to compute outputs for neurons at next layer.
def FeedForward(mlp, inVect, matrix, ActivationFunction, dropout):
	output = ActivationFunction(np.inner(inVect, matrix), mlp)
	# Perform dropout, skipping the last bias neuron
	nNeuron = len(output) - 1
	nDrop = int(dropout * nNeuron)
	if nDrop > 0:
		indexes = [i for i in range(nNeuron)]
		np.random.shuffle(indexes)
		for i in range(nDrop):
			output[indexes[i]] = 0
	# Add output for the biase neuron, always 1
	output[-1] = 1
	return output

# Inference MLP and return a list of output vectors for each layer, including the input layer.
def InferenceMLP(mlp, inVect, dropout):
	results = [inVect]
	weights = mlp['weights']
	n = len(weights)
	# Inference hidden layers.
	for i in range(n - 1):
		outVect = FeedForward(mlp, inVect, weights[i], mlp['actfunc'], dropout)
		results.append(outVect)
		inVect = outVect
	# Inference the output layer.
	outVect = FeedForward(mlp, inVect, weights[n-1], mlp['outfunc'], 0)
	results.append(outVect)
	return results

# Train MLP with an example and get the gradient for each weight.
def TrainOneExample(mlp, train, target, dropout):
	# +[1] for the bias neuron.
	train = np.array(train + [1])
	target = np.array(target + [1])
	# Feedforward: calc the neuron outputs for each layer.
	allOutputs = InferenceMLP(mlp, train, dropout)
	nLayer = len(allOutputs)
	allDeltas = []
	# Calc the error of this training example.
	output = allOutputs[-1]
	error = CalcError(target[:-1], output[:-1])
	# Backward: Calc delta at the output layer. 
	# See 'https://www.ics.uci.edu/~pjsadows/notes.pdf' for using Softmax activation function.
	n = len(output)
	DerivativeFunction = mlp['derivative']
	delta = output - target
	if mlp['outfunc'] not in (Softmax, Linear): 
		delta = delta * DerivativeFunction(output, mlp)
	allDeltas = [delta]
	# Backward: Calc delta for the hidden layers.
	weights = mlp['weights']
	for k in reversed(range(1, nLayer - 1)):
		output = allOutputs[k]
		delta = np.inner(delta, weights[k].transpose())
		delta = delta * DerivativeFunction(output, mlp)
		allDeltas.insert(0, delta)
	# Finally add a dummy list for delta of the input layer for code simplicity.
	allDeltas.insert(0, [])
	# Calc gradients for all weights.
	gradients = []
	for k in range(len(weights)):
		output = allOutputs[k]
		delta = allDeltas[k+1]
		gradients.append(np.outer(delta, output))
	# Return inference error and gradient w.r.t. all weights.
	return gradients, error

# Train MLP with a batch of examples. This function can also be called by reinforcement learning.
def TrainMLP_Batch(mlp, batchdata, alpha, dropout = 0):
	# If global ADAM has not been initialized, do it.
	if not ADAM:
		InitADAM(mlp['layers'])
	# Go through the batch data set to accumulate delta gradients.
	error = 0
	aweights = None
	for train in batchdata:
		dweights, err = TrainOneExample(mlp, train['train'], train['target'], dropout)
		error += err
		aweights = AddWeights(aweights, dweights)
	# Average the accvumulated delta gradients, and update MLP weights with ADAM.
	if len(batchdata) > 1:
		aweights = ScaleWeights(aweights, 1.0/len(batchdata))
	aweights = AdamOptimization(ADAM, aweights, alpha)
	mlp['weights'] = AddWeights(mlp['weights'], aweights)
	# Return inference error of the batch data set.
	return error

def AdamOptimization(adam, gradients, alpha):
	# Get Adam parameters.
	beta1, beta1t = adam['beta1'], adam['beta1t']
	beta2, beta2t = adam['beta2'], adam['beta2t']
	mt, vt = adam['mt'], adam['vt']
	epsilon = adam['epsilon']
	# Perform Adam optimization for all weights. (https://arxiv.org/pdf/1412.6980.pdf)
	adgrad = []
	for k in range(len(gradients)):
		g = gradients[k]
		mt[k] = beta1 * mt[k] + (1 - beta1) * g
		vt[k] = beta2 * vt[k] + (1 - beta2) * g * g
		beta2t_p = np.sqrt(1 - beta2t)
		epsilon_p = epsilon * beta2t_p 
		alpha_p = alpha * beta2t_p / (1 - beta1t)
		adgrad.append(-alpha_p * mt[k] / (np.sqrt(vt[k]) + epsilon_p))
	# Update Adam's beta coefficients.
	adam['beta1t'] *= adam['beta1']
	adam['beta2t'] *= adam['beta2']
	return adgrad

def ScaleWeights(weights, scale):
	sweights = []
	for k in range(len(weights)):
		sweights.append(weights[k] * scale)
	return sweights

def AddWeights(weights, dweights):
	if not weights:
		aweights = dweights
	else:
		aweights = []
		for k in range(len(weights)):
			aweights.append(weights[k] + dweights[k])
	return aweights

# MLP training the whole data set. Update MLP weights and return a list of errors of each epoch. 
def TrainMLP(mlp, nEpoch, nMiniBatch, alpha, lrDecayStep, lrDecayFactor, dropout, modelfile, verbosely):
	allError = []
	nDecayCountDown = lrDecayStep
	for epoch in range(nEpoch):
		# Randomly shuffle the training data.
		np.random.shuffle(TRAIN)
		# Loop to go through all the minibatches of training data.
		error = 0
		for i in range(0, len(TRAIN), nMiniBatch):
			minibatch = TRAIN[i:i+nMiniBatch]
			error += TrainMLP_Batch(mlp, minibatch, alpha, dropout)
		# Decay learning rate when a step size of epochs has been reached.
		nDecayCountDown -= 1
		if nDecayCountDown == 0:
			alpha = alpha * lrDecayFactor
			nDecayCountDown = lrDecayStep
		# Collect error for the current epoch.
		allError.append(error)
		# Save current model whenever 10% progress is made.
		nBlk = (nEpoch // 10) if nEpoch >= 10 else 1
		if epoch > 0 and epoch % nBlk == 0:
			SaveMLP(mlp, modelfile)
		# Display training information of current epoch.
		if verbosely:
			print('epoch=%d error=%.6f'%(epoch, error))
			sys.stdout.flush()
	return allError

# Open UTF8/UTF16 text file, assuming UTF16 has b'\xff\xfe' BOM, and skip BOM before return.
def OpenUnicodeFile(filename):
	fp = open(filename, 'rb')
	if fp.read(2) == b'\xff\xfe':
		fp.close()
		fp = open(filename, encoding = 'utf-16-le')
		fp.seek(2) # skip 2-byte UTF16 BOM
	else:
		fp.seek(0, 0)
		utf8BOM = True if fp.read(3) == b'\xef\xbb\xbf' else False
		fp.close()
		fp = open(filename, encoding = 'utf-8')
		if utf8BOM:	fp.seek(3) # skip 3-byte UTF8 BOM
	return fp

# Load the training data and targets from the input file to TRAIN[].
def LoadTrainingData(filename):
	fp = OpenUnicodeFile(filename)
	nDuplicate = 0
	for line in fp:
		line = line.strip()
		if not line or line[0] == '#':
			continue
		texts = list(filter(None, line.split('\t')))
		if len(texts) < 2:
			continue
		# Splitting to get training example and the target. +[1] for bias neuron.
		texts[0] = texts[0].strip()
		texts[1] = texts[1].strip()
		train  = [float(s) for s in texts[0].split()]
		target = [float(s) for s in texts[1].split()]
		# Add it to training set only if it is not a duplicate.
		if any(x for x in TRAIN if x['train'] == train):
			nDuplicate += 1
		else:
			TRAIN.append({'train': train, 'target': target})
	return nDuplicate

# Load MLP from a previously pickled file.
def LoadMLP(filename): 
	mlp = pickle.load(open(filename, 'rb'))
	return InitMLP(mlp['layers'], mlp['actfuncname'], mlp['outfuncname'], mlp)

# Use pickle to save the MLP to file. Duplicate it with dict(), since some values in MLP will be modified here.
def SaveMLP(mlp, filename):
	mlp = dict(mlp)
	mlp['actfunc'] = mlp['derivative'] = mlp['outfunc'] = None
	fp = open(filename, 'wb')
	pickle.dump(mlp, fp)
	fp.close()

# Make a copy, instead of a reference, of the MLP.
def DuplicateMLP(mlp):
	mlp = dict(mlp)
	mlp['actfunc'] = mlp['derivative'] = mlp['outfunc'] = None
	mlp = pickle.loads(pickle.dumps(mlp))
	return InitMLP(mlp['layers'], mlp['actfuncname'], mlp['outfuncname'], mlp)

def main():
	parser = argparse.ArgumentParser(description = 'Multilayer Perceptron training tool.')
	parser.add_argument("TrainingData", help = "text training data file")
	parser.add_argument("Model", help = "output binary NN model.")
	parser.add_argument("-im", "--initmodel", help = "optional initial NN model.")
	parser.add_argument("-tp", "--topology", help = "number of neurons in each layer, example '6-8-3', required if --initmodel is missing.")
	parser.add_argument("-ep", "--epoch", help = "number of epochs, default 1000.")
	parser.add_argument("-mb", "--minibatch", help = "mini batch size, default 1 for stochastic gradient descent.")
	parser.add_argument("-lr", "--learnrate", help = "learning rate alpha, default 0.1.")
	parser.add_argument("-lrdf", "--learnratedecayfactor", help = "learning rate decay factor, default 0.9.")
	parser.add_argument("-lrds", "--learnratedecaystep", help = "steps to decay the learning rate, default 50 steps.")
	parser.add_argument("-af", "--actfunc", help = "activation function in hidden layers, 'sigmoid'(default), 'tanh', or 'relu'.")
	parser.add_argument("-of", "--outfunc", help = "output function, 'linear' or 'softmax', default is the function used in hidden layer.")
	parser.add_argument("-dp", "--dropout", help = "drop outputs of a portion (0~1) of neurons, default 0, no dropout.")
	parser.add_argument("--noplot", help = "do not plot error chart", action = "store_true")
	parser.add_argument("-v", "--verbosely", help = "verbosely showing the training progress", action = "store_true")
	args = parser.parse_args()

	# Load or create an initial model.
	if args.initmodel:
		mlp = LoadMLP(args.initmodel)
	elif args.topology:
		layers = [int(s) for s in args.topology.split('-')]
		actfunc = args.actfunc if args.actfunc else 'sigmoid'
		outfunc = args.outfunc if args.outfunc else actfunc
		mlp = InitMLP(layers, actfunc, outfunc)
	else:
		raise Exception("Neither '--initmodel/-im' nor '--topology/-tp' is found.")

	# Determine epochs, minibatch, learning rate alpha, alpha's decay factor and step size, and dropout percent.
	nEpoch = max(1, (int(args.epoch))) if args.epoch else 1000
	nMiniBatch = max(1, (int(args.minibatch))) if args.minibatch else 1
	alpha = max(0, float(args.learnrate)) if args.learnrate else 0.1
	lrDecayStep = max(1,  nEpoch // int(args.learnratedecaystep)) if args.learnratedecaystep else max(1, nEpoch // 50)
	lrDecayFactor = max(0.1, min(1, float(args.learnratedecayfactor))) if args.learnratedecayfactor else 0.9
	dropout = max(0, min(1, float(args.dropout))) if args.dropout else 0

	# Load training data to global list TRAIN[].
	nDup = LoadTrainingData(args.TrainingData)
	if nDup > 0:
		print('Warning: %d training data are duplicate and discarded.\n'%nDup)

	# Display MLP information and training parameters.
	if args.verbosely:
		print('topology:', mlp['layers'])
		print('hidden activation:', mlp['actfunc'].__name__)
		print('output activation:', mlp['outfunc'].__name__)
		print('learning rate: %f (%d, %.2f)'%(alpha, lrDecayStep, lrDecayFactor))
		print('dropout: %d%%'%int(dropout*100))
		print('epochs: %d (mb=%d)\n'%(nEpoch, nMiniBatch))
		sys.stdout.flush()

	# Training function, return a list of errors.
	error = TrainMLP(mlp, nEpoch, nMiniBatch, alpha, lrDecayStep, lrDecayFactor, dropout, args.Model, args.verbosely)

	# Output trained model.
	SaveMLP(mlp, args.Model)

	# Draw the error chart. 
	if not args.noplot:
		title = 'Training %s error: %.4f\n'%(args.Model, error[-1])
		title += 'topology: %s, '%('-'.join(str(n) for n in mlp['layers']))
		title += 'hidden: %s'%(mlp['actfunc'].__name__)
		if mlp['outfunc'] != mlp['actfunc']:
			title += ', out: %s'%(mlp['outfunc'].__name__)
		plt.title(title, fontsize = 12)
		xlabel = 'mb: %d, '%nMiniBatch
		xlabel += 'alpha: %f (%d, %.2f), '%(alpha, lrDecayStep, lrDecayFactor)
		xlabel += 'dropout: %d%%'%int(dropout * 100)
		plt.xlabel('epochs (' + xlabel + ')', fontsize = 12)
		plt.ylabel('error', fontsize = 12)
		plt.plot(error)
		plt.show()

if __name__ == '__main__':
	main()
