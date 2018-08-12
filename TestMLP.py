# TestMLP.py
# Author: Taihsuan Ho, 2018/06/07
# A multilayer perceptron testing program.

import argparse
import pickle
import numpy as np

def Sigmoid(v, *_):
	return 1 / (1 + np.exp(-v))

def Tanh(v, *_):
	return 2 / (1 + np.exp(-2*v)) - 1

def ReLU(v, mlp):
	return np.array([x if x >= 0 else 0 for x in v])

def Linear(v, *_):
	return np.array(v)

def Softmax(v, *_):
	ev = np.exp(v - np.max(v))
	return ev / ev.sum()

def FeedForward(mlp, inVect, matrix, ActivationFunction):
	# Feed forward to compute outputs for neurons at next layer. Skip the last bias neuron, which outputs 1 always.
	output = ActivationFunction(np.inner(inVect, matrix[:-1]), mlp)
	output = np.append(output, [1])
	return output

def InferenceMLP(mlp, inVect):
	# Inference MLP and return a list of output vectors for each layer, including the input layer.
	inVect = np.array(inVect + [1])
	weights = mlp['weights']
	n = len(weights)
	# Inference hidden layers.
	for i in range(n - 1):
		outVect = FeedForward(mlp, inVect, weights[i], mlp['actfunc'])
		inVect = outVect
	# Inference the output layer.
	outVect = FeedForward(mlp, inVect, weights[n-1], mlp['outfunc'])
	return outVect[:-1].tolist()

def LoadMLP(filename):
	FUNCS = {'tanh':Tanh, 'sigmoid':Sigmoid, 'relu':ReLU, 'linear': Linear, 'softmax': Softmax}
	mlp = pickle.load(open(filename, 'rb'))
	mlp['actfunc'] = FUNCS[mlp['actfuncname']]
	mlp['outfunc'] = FUNCS[mlp['outfuncname']]
	return mlp

'''
# Below codes are for the command line tool.
'''
def OpenUnicodeFile(filename):
	# Open UTF8/UTF16 text file, assuming UTF16 has b'\xff\xfe' BOM, and skip BOM before return.
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

def FormatOutputList(output, fmt):
	s = '['
	for x in output:
		s += fmt%x + ', '
	return s.strip(', ') + ']'

def CalcError(ref, hyp):
	ref = np.array(ref)
	hyp = np.array(hyp)
	err = ref - hyp
	return (err * err).sum()

def main():
	parser = argparse.ArgumentParser(description = 'Multilayer Perceptron testing tool.')
	parser.add_argument("Model", help = "output binary NN model.")
	parser.add_argument("TestData", help = "text test data file")
	args = parser.parse_args()

	# Load NN model.
	mlp = LoadMLP(args.Model)
	print('test:', args.TestData)
	print('model:', args.Model)
	print('topology:', mlp['layers'])
	print('hidden activation:', mlp['actfunc'].__name__)
	print('output activation:', mlp['outfunc'].__name__)
	print()

	# Testing NN
	totError = 0
	fp = OpenUnicodeFile(args.TestData)
	for line in fp:
		line = line.strip()
		if not line or line[0] == '#':
			continue
		texts = list(filter(None, line.split('\t')))
		if len(texts) < 2:
			continue
		test  = [float(s) for s in texts[0].strip().split()]
		target = [float(s) for s in texts[1].strip().split()]
		output = InferenceMLP(mlp, test)
		err = CalcError(target, output)
		totError += err
		output = FormatOutputList(output, '%.2f')
		print(test, target, '=>', output, 'err=%.3f'%err)
	print('TotalError=%f'%totError)

if __name__ == '__main__':
	main()
