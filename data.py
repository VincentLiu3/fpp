import numpy as np
from cycleworld import CycleWorld
from stochastic_data import gen_data
from keras.preprocessing.text import Tokenizer
import os
from reader import _build_vocab, _file_to_word_ids 
# import chainer

def one_hot_encode(sequence, n_unique=10):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return np.array(encoding)
 
# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [np.argmax(vector) for vector in encoded_seq]

def generate_cpy(T,batch_size):
	input_data = np.zeros(shape=[batch_size,T+20,10])
	output_data = np.zeros(shape = [batch_size,T+20,10])
	for i in range(batch_size):
		init = np.random.randint(1,9,size=10)
		blank_middle = np.zeros(T-1)
		special = [9]
		blank_end = np.zeros(10)

		X = one_hot_encode(np.array(np.concatenate((init,blank_middle,special,blank_end),axis = 0),dtype = int))
		Y = one_hot_encode(np.array(np.concatenate((blank_end,blank_middle,special,init),axis = 0),dtype = int))

		input_data[i] = X
		output_data[i] = Y

	return input_data,output_data

def generate_stochastic_data(batch_size, num_batches):
	total_series_length = num_batches * batch_size
	x,y = gen_data(total_series_length)
	X = np.zeros(shape=(batch_size,num_batches,2))
	Y = np.zeros(shape=(batch_size,num_batches,2))
	for i in range(batch_size):
		X[i] = one_hot_encode(np.squeeze(x[i*(num_batches):(i+1)*num_batches]),2)
		Y[i] = one_hot_encode(np.squeeze(y[i*(num_batches):(i+1)*num_batches]),2)
		
	return X,Y



def generate_cw(cycleworld_size, batch_size, num_batches):
    total_series_length = num_batches * batch_size
    c = CycleWorld(cycleworld_size)
    x = np.zeros(shape=(total_series_length,1),dtype=int)
    y = np.zeros(shape=(total_series_length,1),dtype=int)
    for i in range(total_series_length):
        s = c.step(np.zeros(0))
        x[i] = s[2]
        y[i] = s[3]

    X = np.zeros(shape=(batch_size,num_batches,2))
    Y = np.zeros(shape=(batch_size,num_batches,2))

   
    for i in range(batch_size):
    	X[i] = one_hot_encode(np.squeeze(x[i*(num_batches):(i+1)*num_batches]),2)
    	Y[i] = one_hot_encode(np.squeeze(y[i*(num_batches):(i+1)*num_batches]),2)
    
    return (X, Y)


def read_file(file_path):
    """
    Read file, replace newlines with end-of-sentence marker.
    :param filename:
    :return the file as a string
    """

    with open(file_path, 'r') as f:
        return f.read().replace('\n', '<eos>')


def ptb_iterator(raw_data, batch_size):
    """
    Generates generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.
    :param raw_data: one of the raw data outputs from ptb_raw_data.
    :param batch_size: int, the batch size.
    :param num_steps: int, the number of unrolls.
    :return Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the
        right by one.
    :raises ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    # epoch_size = (batch_len - 1) // num_steps

    # if epoch_size == 0:
    #     raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(batch_len-1):
        x = data[:, i]
        y = data[:, i+1]
        yield x,y

def data_ptb(batch_size):
	train_data,_,_,vocab_size,_ = load_data()
	# print(vocab_size)
	X = []
	Y = []
	for step, (x, y) in enumerate(ptb_iterator(train_data, batch_size)):
		X.append(x)
		Y.append(y)

	X = np.reshape(X,[len(X),batch_size,1])
	Y = np.reshape(Y,[len(Y),batch_size,1])

	return X,Y


def load_data():
    # get the data paths
    data_path = 'data/data/'
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    # print(train_data[:5])
    # print(word_to_id)
    # print(vocabulary)
    # print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

def batch_producer(raw_data, batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y

class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)


if __name__ == "__main__":
	
	# for i in range(len(train_data)):
	# 	print(train_data[i])
	# 	input()
	X,Y = data_ptb(20)

	for i in range(5,len(X)):
		print(X[i-5:i])
		print(Y[i-5:i])
		input()
	print(np.shape(X))
	
	# x,y = data_ptb(vocab_size = 10000, batch_size=100, num_steps=10)
	# print(np.shape(x))
	# x,y = data_ptb(vocab_size = 10000,batch_size=3,num_steps=1)
	# # # print(np.shape(x))
	# for iter in range(2,np.shape(x)[0]):
	# 	print(np.reshape(x[iter-2:iter],[3,2]))
	# 	print(np.reshape(y[iter-2:iter],[3,2]))
	# 	input()
	# raw_data = reader.ptb_raw_data('data/data')
	# train_data, valid_data, test_data, _ = raw_data
	# train_input = PTBInput(batch_size=20, num_steps=5, data=train_data, name="TrainInput")
	# print(train_input.input_data)

	# x,y = generate_stochastic_data(3,10)
	# print(np.argmax(x,2))
	# print(np.argmax(y,2))
	# batch_size = 2
	# num_batches = 10
	# X,Y = generate_cw(6,batch_size,num_batches)
	# iter = 0
	# time_steps = 8
	# while iter<num_batches:
	# 	batch_x = X[:,iter:iter+time_steps]
	# 	batch_y = Y[:,iter:iter+time_steps]
	# 	print('start:',iter,'end:',iter+time_steps)
	# 	print(np.shape(np.argmax(batch_x,2)))
	# 	print(np.argmax(batch_x,2))
	# 	print(np.argmax(batch_y,2))
	# 	input()
	# 	iter+=1

	# # batch_no = 1
	# # item = 6
	# # print(x[batch_no-1, item-1])
	# # print(y[batch_no-1, item-1])
	# batch_size = 200
	# num_batches = 2
	# x,y = generate_cpy(batch_size-20,num_batches)
	# print(np.shape(x))
	# print(np.argmax(x,axis = 2))
	# print(np.argmax(y,axis = 2))





