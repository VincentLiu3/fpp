import numpy as np
import tensorflow as tf


class BPTT_Model:
	def __init__(self,
				 dataset,
				 use_lstm,
				 n_input,
				 n_classes,
				 num_units,
				 batch_size,
				 time_steps,
				 learning_rate,
				 clip_gradients,
				 run_number):

		tf.set_random_seed(run_number)
		self.time_steps = time_steps
		self.learning_rate = learning_rate
		self.num_units = num_units
		self.n_classes = n_classes
		self.use_lstm = use_lstm
		self.clip_gradients = clip_gradients
		self.dataset = dataset
		self.batch_size = batch_size
		if self.dataset == 'ptb':
			self.x = tf.placeholder(tf.int32, shape=[None, time_steps])
			self.y = tf.placeholder(tf.int32, shape=[None, time_steps])
			self.vocab_size = 10000
			self.n_classes = self.vocab_size
			init_scale = 0.1
			self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.num_units], -init_scale, init_scale))
		else:
			self.x = tf.placeholder(tf.float32, shape=[None, time_steps, n_input])
			self.y = tf.placeholder(tf.float32, shape=[None, time_steps, n_classes])
			self.sequentialise_input()
		
		self.setup_model()

		self.create_output()
		self.create_loss()

		self.create_train()


	def setup_model(self):
		if self.dataset == 'ptb':
			self.cell = tf.nn.rnn_cell.BasicRNNCell(self.num_units)
			# self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=0.5)
			# self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * 2, state_is_tuple=True)
			self.state_placeholder = tf.placeholder(tf.float32, shape = [2, None, self.num_units])
			# l = tf.unstack(self.state_placeholder, axis=0)
			# self.init_state = tf.nn.rnn_cell.LSTMStateTuple(l[0],l[1])
			self.init_state = self.state_placeholder
		if self.use_lstm:
			self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
			self.state_placeholder = tf.placeholder(tf.float32, shape = [2, None, self.num_units])
			l = tf.unstack(self.state_placeholder, axis=0)
			self.init_state = tf.nn.rnn_cell.LSTMStateTuple(l[0],l[1])
		else:
			self.cell = tf.nn.rnn_cell.BasicRNNCell(self.num_units)
			self.state_placeholder = tf.placeholder(tf.float32, shape = [None, self.num_units])
			self.init_state = self.state_placeholder

		self.W = tf.get_variable('W', [self.num_units, self.n_classes])
		self.b = tf.get_variable('b', [self.n_classes], initializer=tf.constant_initializer(0.0))


	def sequentialise_input(self):
		self.x_sequential = tf.unstack(self.x, self.time_steps, 1)

	def create_output(self):
		if self.dataset == 'ptb':
			inputs = tf.nn.embedding_lookup(self.embedding, self.x)
			inputs = tf.reshape(inputs, [self.time_steps,self.batch_size,self.num_units])
			inputs = tf.unstack(inputs, self.time_steps, 0)			

			output, self.state_last = tf.nn.static_rnn(self.cell, 
												   inputs,
												   initial_state = self.init_state,
												   dtype=tf.float32)
			# reshape to (batch_size * num_steps, hidden_size)
			self.lstm_output = tf.reshape(output, [-1, self.num_units])
			self.logits = tf.nn.xw_plus_b(self.lstm_output, self.W, self.b)
			self.logits = tf.reshape(self.logits, [self.batch_size, self.time_steps, self.vocab_size])
			self.output = tf.nn.softmax(self.logits)

			if self.use_lstm:
				self.state = self.state_last
				
				_, self.state_first = tf.nn.static_rnn(self.cell, 
														   [inputs[0]],
														   initial_state = self.init_state,
														   dtype=tf.float32)
			else:
				# self.state_last = self.lstm_output[-1]
				self.state_first = self.lstm_output[0]
				self.state = self.state_last

		else:
			self.lstm_output, self.state_last = tf.nn.static_rnn(self.cell, 
													   self.x_sequential,
													   initial_state = self.init_state,
													   dtype=tf.float32)
			self.logits = [tf.matmul(output_t, self.W) + self.b for output_t in self.lstm_output]
			self.output = [tf.nn.softmax(logit) for logit in self.logits]

			if self.use_lstm:
				self.state = self.state_last
				
				_, self.state_first = tf.nn.static_rnn(self.cell, 
														   [self.x_sequential[0]],
														   initial_state = self.init_state,
														   dtype=tf.float32)
			else:
				self.state_last = self.lstm_output[-1]
				self.state_first = self.lstm_output[0]
				self.state = self.state_first

		if self.dataset == 'ptb':
			self.accuracy = tf.constant(0.0)
		else:
			self.correct_prediction = tf.equal(tf.argmax(self.output[-1],1),tf.transpose(tf.argmax(self.y[:,-1,:],1)))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))


	def create_loss(self):
		if self.dataset == 'ptb':
			loss = tf.contrib.seq2seq.sequence_loss(
			            tf.reshape(self.logits[:,-1,:],[self.batch_size,1,tf.shape(self.logits)[-1]]),
			            tf.reshape(self.y[:,-1],[self.batch_size,1]),
			            tf.ones([self.batch_size, 1], dtype=tf.float32),
			            average_across_timesteps=False,
			            average_across_batch=True)
			# Update the cost
			self.loss = tf.reduce_sum(loss)
		else:
			self.y_as_list = tf.unstack(self.y, num=self.time_steps, axis=1)
			self.loss = tf.losses.softmax_cross_entropy(self.y[:,-1],self.logits[-1])
			# self.loss = tf.reduce_mean([tf.losses.softmax_cross_entropy(label,logit) for \
			#           logit, label in zip(self.logits, self.y_as_list)])

	def create_train(self):
		optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
		self.gvs = optimizer.compute_gradients(self.loss)
		if self.clip_gradients:
			capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in self.gvs]
			self.train = optimizer.apply_gradients(capped_gvs)
		else:
			self.train = optimizer.apply_gradients(self.gvs)
