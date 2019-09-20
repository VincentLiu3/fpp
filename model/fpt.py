# import numpy as np
import tensorflow as tf


class FPT_Model:
    def __init__(self, dataset, use_buffer_bptt, fix_buffer, n_input, n_classes, num_units, batch_size, sample_updates,
                 learning_rate, clip_gradients, lambda_state, run_number):
        assert dataset != 'ptb'
        tf.set_random_seed(run_number)
        self.use_buffer_bptt = use_buffer_bptt
        self.fix_buffer = fix_buffer
        self.learning_rate = learning_rate
        self.state_learning_rate = learning_rate
        self.learning_rate_output = learning_rate
        self.num_units = num_units  # number of hidden state in RNN
        self.n_input = n_input  # inpute dimension
        self.n_classes = n_classes
        self.clip_gradients = clip_gradients
        self.sample_updates = sample_updates  # T = updates_per_step
        self.batch_size = batch_size
        self.lambda_state = lambda_state
        self.dataset = dataset

        self.num_update = 1  # number of blocks to sample for each time step

        self.x = tf.placeholder(tf.float32, shape=[1, None, n_input])  # [T, 1, n_input]
        self.y = tf.placeholder(tf.float32, shape=[None, n_classes])
        self.x_sequential = tf.unstack(self.x, axis=0)  # a list of [1, n_input]

        # self.setup_model()
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
            self.cell = tf.nn.rnn_cell.BasicRNNCell(self.num_units, reuse=tf.get_variable_scope().reuse)
            self.init_state = tf.placeholder(tf.float32, shape=[1, self.num_units])

            self.W = tf.get_variable('W', [self.num_units, self.n_classes],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable('b', [self.n_classes], initializer=tf.constant_initializer(0.0))

        # self.create_output()
        self.lstm_output, self.state = tf.nn.static_rnn(self.cell, self.x_sequential,
                                                        initial_state=self.init_state, dtype=tf.float32)
        # self.state.shape = [1, n_units]
        self.logits = [tf.matmul(output_t, self.W) + self.b for output_t in self.lstm_output]
        self.output = [tf.nn.softmax(logit) for logit in self.logits]

        self.correct_prediction = tf.equal(tf.argmax(self.output, 2), tf.transpose(tf.argmax(self.y, 1)))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

        # self.current_loss()
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y, self.logits[-1]))

        # self.get_training_vars_batch()
        self.x_t = tf.placeholder(tf.float32, [self.sample_updates, None, self.n_input], name='Input0')
        self.y_t = tf.placeholder(tf.float32, [self.sample_updates, None, self.n_classes], name='Target')
        self.s_tm1 = tf.placeholder(tf.float32, [self.sample_updates, None, self.num_units], name='State_tm1')
        self.s_t = tf.placeholder(tf.float32, [self.sample_updates, None, self.num_units], name='State_t')

        # self.traning_loss_batch()
        # self.create_train()

        # self.traning_loss_sequential()
        state = self.s_tm1[0]
        state_t = self.s_t[-1]

        # self.x_new_seq = tf.squeeze(self.x_t, [2])  # remove all dimensions of size 1 (only axis 2 here)
        self.input_t_seq = tf.unstack(self.x_t, self.sample_updates, axis=0)  # (1, batch_size, n_input)
        self.lstm_outputs_t_seq, self.state_t_seq = tf.nn.static_rnn(self.cell, self.input_t_seq,
                                                                     dtype="float32", initial_state=state)
        # self.lstm_outputs_t_seq: a list of [None, n_unit]
        self.output_t_logits_seq = tf.matmul(self.lstm_outputs_t_seq[-1], self.W) + self.b  # [None, n_class]
        self.loss_state_seq = self.lambda_state * tf.losses.mean_squared_error(state_t, self.state_t_seq)

        self.y_as_list_seq = tf.unstack(self.y_t, num=self.sample_updates, axis=0)  # a list of [None, n_class]
        self.loss_target_seq = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y_t[-1], self.output_t_logits_seq))

        self.pred_loss_seq = tf.equal(tf.argmax(self.y_t[-1], 1), tf.argmax(self.output_t_logits_seq, 1))
        self.pred_acc = tf.reduce_mean(tf.cast(self.pred_loss_seq, "float"))

        if self.use_buffer_bptt:
            # for buffer-BPTT, update the parameters according to the loss.
            self.total_loss_seq = tf.reduce_mean(self.loss_target_seq)

            if self.fix_buffer:
                state_t = self.state_t_seq
        else:
            self.total_loss_seq = tf.reduce_mean(tf.add(self.loss_target_seq, self.loss_state_seq))
            grad = tf.gradients(self.total_loss_seq, state)
            state -= tf.squeeze(tf.multiply(self.state_learning_rate, grad), axis=0)
            grad = tf.gradients(self.total_loss_seq, state_t)
            state_t -= tf.squeeze(tf.multiply(self.state_learning_rate, grad), axis=0)

        self.state_tm1_c_seq = state
        self.state_t_c_seq = state_t

        # self.create_train_sequential()
        lstm_variables = [v for v in tf.trainable_variables() if v.name.startswith('rnn')]
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            optimizer_lstm = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            gvs_lstm = optimizer_lstm.compute_gradients(self.total_loss_seq, var_list=lstm_variables)
            if self.clip_gradients:
                capped_gvs_lstm = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs_lstm]
                self.train_lstm = optimizer_lstm.apply_gradients(capped_gvs_lstm)
            else:
                self.train_lstm = optimizer_lstm.apply_gradients(gvs_lstm)

            optimizer_output = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_output)
            gvs_output = optimizer_output.compute_gradients(self.total_loss_seq, var_list=[self.W, self.b])
            if self.clip_gradients:
                capped_gvs_output = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs_output]
                self.train_output = optimizer_output.apply_gradients(capped_gvs_output)
            else:
                self.train_output = optimizer_output.apply_gradients(gvs_output)

        self.train_seq = tf.group(self.train_lstm, self.train_output)

        # Validation
        self.val_input = tf.placeholder(tf.float32, [self.sample_updates, 1, self.n_input], name='Input0')
        self.val_label = tf.placeholder(tf.float32, [self.sample_updates, 1, self.n_classes], name='Target')

        self.input_seq = tf.unstack(self.val_input, self.sample_updates, axis=0)  # (M, T, batch_size, n_inpu)
        self.output_seq, self.val_state = tf.nn.static_rnn(self.cell, self.input_seq, dtype="float32", initial_state=self.init_state)

        if tf.__version__ == '1.14.0':
            self.logits_seq = tf.matmul(self.output_seq, self.W) + self.b

            # self.y_as_list_seq = tf.unstack(self.y_t, num=self.sample_updates, axis=0)
            self.val_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.val_label, self.logits_seq))
            self.val_pred = tf.equal(tf.argmax(self.val_label, 2), tf.argmax(self.logits_seq, 2))
            self.val_acc = tf.reduce_mean(tf.cast(self.val_pred, "float"))
        else:
            self.output_seq = tf.squeeze(self.output_seq, axis=1)
            self.val_label = tf.squeeze(self.val_label, axis=1)

            self.logits_seq = tf.matmul(self.output_seq, self.W) + self.b

            self.val_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.val_label, self.logits_seq))
            self.val_pred = tf.equal(tf.argmax(self.val_label, 1), tf.argmax(self.logits_seq, 1))
            self.val_acc = tf.reduce_mean(tf.cast(self.val_pred, "float"))

    # def sequentialise_input(self):
    #     assert False, 'why created this function?'

    # def change_sample_updates(self, sample_updates):
    #     self.sample_updates = sample_updates
    #     self.get_training_vars_batch()
    #     # self.traning_loss_batch()
    #     # self.create_train()
    #     self.get_corrected_state_batch()
    #     self.traning_loss_sequential()
    #     self.create_train_sequential()
    #
    #     if self.state_updates != 0:
    #         self.buffer_update_batch()

    # def update_state_lr(self, learning_rate):
    #     self.state_learning_rate = learning_rate
    #
    # def setup_model(self):
    #     with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
    #         if self.dataset == 'ptb':
    #             self.cell = tf.nn.rnn_cell.LSTMCell(self.num_units, state_is_tuple=True, forget_bias=1.0)
    #             self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=0.5)
    #             self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * 2, state_is_tuple=True)
    #             self.state_placeholder = tf.placeholder(tf.float32, shape=[2, self.batch_size, self.num_units])
    #             l = tf.unstack(self.state_placeholder, axis=0)
    #             self.init_state = tf.nn.rnn_cell.LSTMStateTuple(l[0], l[1])
    #
    #         if self.use_lstm:
    #             self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
    #             self.state_placeholder = tf.placeholder(tf.float32, shape=[2, self.batch_size, self.num_units])
    #             l = tf.unstack(self.state_placeholder, axis=0)
    #             self.init_state = tf.nn.rnn_cell.LSTMStateTuple(l[0], l[1])
    #         else:
    #             self.cell = tf.nn.rnn_cell.BasicRNNCell(self.num_units, reuse=tf.get_variable_scope().reuse)
    #             self.state_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_units])
    #             self.init_state = self.state_placeholder
    #
    #         self.W = tf.get_variable('W', [self.num_units, self.n_classes],
    #                                  initializer=tf.contrib.layers.xavier_initializer())
    #         self.b = tf.get_variable('b', [self.n_classes], initializer=tf.constant_initializer(0.0))
    #
    # def create_output(self):
    #     if self.dataset == 'ptb':
    #         inputs = tf.nn.embedding_lookup(self.embedding, self.x)
    #         inputs = tf.unstack(inputs, axis = 1)
    #
    #         output, self.state = tf.nn.static_rnn(self.cell,
    #                                                inputs,
    #                                                initial_state = self.init_state,
    #                                                dtype=tf.float32)
    #         # reshape to (batch_size * num_steps, hidden_size)
    #         self.lstm_output = tf.reshape(output, [-1, self.num_units])
    #         self.logits = tf.nn.xw_plus_b(self.lstm_output, self.W, self.b)
    #         self.logits = tf.reshape(self.logits, [self.batch_size, 1, self.vocab_size])
    #         self.output = tf.nn.softmax(self.logits)
    #         self.accuracy = tf.constant(0.0)
    #     else:
    #         self.lstm_output, self.state = tf.nn.static_rnn(self.cell, self.x_sequential,
    #                                                         initial_state = self.init_state, dtype=tf.float32)
    #
    #         self.logits = [tf.matmul(output_t, self.W) + self.b for output_t in self.lstm_output]
    #         self.output = [tf.nn.softmax(logit) for logit in self.logits]
    #
    #         self.correct_prediction = tf.equal(tf.argmax(self.output,2),tf.transpose(tf.argmax(self.y,1)))
    #         self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
    #
    # def current_loss(self):
    #     if self.dataset == 'ptb':
    #         loss = tf.contrib.seq2seq.sequence_loss(
    #                     self.logits,
    #                     self.y,
    #                     tf.ones([self.batch_size, 1], dtype=tf.float32),
    #                     average_across_timesteps=False,
    #                     average_across_batch=True)
    #         # Update the cost
    #         self.loss = tf.reduce_sum(loss)
    #     else:
    #         self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y, self.logits[-1]))
    #
    # def current_train(self):
    #     with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
    #         optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    #         self.gvs = optimizer.compute_gradients(self.loss)
    #         if self.clip_gradients:
    #             capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in self.gvs]
    #             self.train_current = optimizer.apply_gradients(capped_gvs)
    #         else:
    #             self.train_current = optimizer.apply_gradients(self.gvs)
    #
    # def get_training_vars_batch(self):
    #     if self.dataset == 'ptb':
    #         self.x_t = tf.placeholder(tf.int32, [self.sample_updates, self.batch_size, 1], name='Input0')
    #         self.y_t = tf.placeholder(tf.int32, [self.sample_updates, self.batch_size,1], name='Target')
    #     else:
    #         self.x_t = tf.placeholder(tf.float32, [self.sample_updates, self.batch_size, 1, self.n_input], name='Input0')
    #         self.y_t = tf.placeholder(tf.float32, [self.sample_updates, self.batch_size, self.n_classes], name='Target')
    #
    #     if self.use_lstm:
    #         self.s_tm1 = tf.placeholder(tf.float32, [None,2, self.batch_size, self.num_units])
    #         self.s_t = tf.placeholder(tf.float32, [None,2, self.batch_size, self.num_units])
    #         self.state_t=self.s_t
    #     else:
    #         self.s_tm1 = tf.placeholder(tf.float32, [self.sample_updates, self.batch_size, self.num_units], name='State_tm1')
    #         self.state_tm1 = self.s_tm1  # S_(t-1) from the buffer
    #         self.s_t = tf.placeholder(tf.float32, [self.sample_updates, self.batch_size, self.num_units], name='State_t')
    #         self.state_t = self.s_t
    #
    # def traning_loss_batch(self):
    #     self.total_loss = 0.0
    #     state_tm1_list = []
    #     state_t_list = []
    #
    #     for idx in range(self.sample_updates):
    #         if self.dataset == 'ptb':
    #             inputs = tf.nn.embedding_lookup(self.embedding, self.x_t[idx])
    #             self.input_t = tf.unstack(inputs, axis=1)
    #         else:
    #             self.input_t = tf.unstack(self.x_t[idx], axis=1)
    #
    #         if self.dataset == 'ptb':
    #             state = self.s_tm1[idx]
    #             state_t = self.state_t[idx]
    #             # l = tf.unstack(state, axis=0)
    #             # # state_o = l[1]
    #             # state_lstm = tf.nn.rnn_cell.LSTMStateTuple(l[0],l[1])
    #             output, state_t_new = tf.nn.static_rnn( self.cell,
    #                                                self.input_t,
    #                                                initial_state = state,
    #                                                dtype=tf.float32 )
    #             lstm_output = tf.reshape(output, [-1, self.num_units])
    #             logits = tf.nn.xw_plus_b(lstm_output, self.W, self.b)
    #             output_logits = tf.reshape(logits, [self.batch_size, 1, self.vocab_size])
    #
    #             loss_target = tf.contrib.seq2seq.sequence_loss(
    #                     output_logits,
    #                     self.y_t[idx],
    #                     tf.ones([self.batch_size, 1], dtype=tf.float32),
    #                     average_across_timesteps=False,
    #                     average_across_batch=True)
    #             loss_state_t = self.lambda_state*tf.reduce_mean(tf.losses.mean_squared_error(state_t_new,state_t))
    #             total_loss = tf.add(loss_state_t,loss_target)
    #             self.total_loss += total_loss
    #             grad = tf.gradients(total_loss,state)
    #             state -= tf.squeeze(tf.multiply(self.state_learning_rate,grad))
    #             grad = tf.gradients(total_loss,state_t)
    #             state_t -= tf.squeeze(tf.multiply(self.state_learning_rate,grad))
    #             state_tm1_list.append(state)
    #             state_t_list.append(state_t)
    #         else:
    #             if self.use_lstm:
    #                 state = self.s_tm1[idx]
    #                 state_t = self.state_t[idx]
    #                 l = tf.unstack(state, axis=0)
    #                 state_lstm = tf.nn.rnn_cell.LSTMStateTuple(l[0],l[1])
    #                 output,state_t_new=tf.nn.static_rnn(self.cell,self.input_t,dtype="float32",initial_state=state_lstm)
    #                 output_logits = tf.matmul(tf.squeeze(output), self.W) + self.b
    #                 loss_target = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y_t[idx],output_logits))
    #                 loss_state_t = tf.reduce_mean(tf.losses.mean_squared_error(state_t_new,state_t))
    #                 total_loss = tf.add(loss_state_t,loss_target)
    #                 self.total_loss += total_loss
    #                 grad = tf.gradients(total_loss,state)
    #                 state -= tf.squeeze(tf.multiply(self.state_learning_rate,grad))
    #                 grad = tf.gradients(total_loss,state_t)
    #                 state_t -= tf.squeeze(tf.multiply(self.state_learning_rate,grad))
    #                 state_tm1_list.append(state)
    #                 state_t_list.append(state_t)
    #             else:
    #                 state = self.state_tm1[idx]
    #                 state_t = self.state_t[idx]
    #                 _, state_t_new = tf.nn.static_rnn(self.cell, self.input_t, dtype="float32", initial_state=state)
    #                 output_logits = tf.matmul(state_t_new, self.W) + self.b
    #                 loss_target = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y_t[idx], output_logits))
    #                 loss_state_t = tf.reduce_mean(tf.losses.mean_squared_error(state_t_new, state_t))  # mean?
    #                 total_loss = tf.add(loss_state_t, loss_target)
    #                 self.total_loss += total_loss
    #                 grad = tf.gradients(total_loss, state)
    #                 state -= tf.squeeze(tf.multiply(self.state_learning_rate, grad))
    #                 grad = tf.gradients(total_loss, state_t)
    #                 state_t -= tf.squeeze(tf.multiply(self.state_learning_rate, grad))
    #                 state_tm1_list.append(state)
    #                 state_t_list.append(state_t)
    #
    #     self.state_tm1_c = tf.stack(state_tm1_list)
    #     self.state_t_c = tf.stack(state_t_list)
    #
    # def create_train(self):
    #     lstm_variables = [v for v in tf.trainable_variables() if v.name.startswith('rnn')]
    #     with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    #         optimizer_lstm = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    #         gvs_lstm = optimizer_lstm.compute_gradients(self.total_loss, var_list=lstm_variables)
    #         if self.clip_gradients:
    #             capped_gvs_lstm = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs_lstm]
    #             self.train_lstm = optimizer_lstm.apply_gradients(capped_gvs_lstm)
    #         else:
    #             self.train_lstm = optimizer_lstm.apply_gradients(gvs_lstm)
    #
    #         optimizer_output = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_output)
    #         gvs_output = optimizer_output.compute_gradients(self.total_loss, var_list=[self.W, self.b])
    #         if self.clip_gradients:
    #             capped_gvs_output = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs_output]
    #             self.train_output = optimizer_output.apply_gradients(capped_gvs_output)
    #         else:
    #             self.train_output = optimizer_output.apply_gradients(gvs_output)
    #
    #     self.train = tf.group(self.train_lstm, self.train_output)
    #
    # def traning_loss_sequential(self):
    #     state = self.s_tm1[0]
    #     state_t = self.s_t[-1]
    #
    #     self.x_new_seq = tf.squeeze(self.x_t, [2])  # remove all dimensions of size 1 (only axis 2 here)
    #     if self.dataset == 'ptb':
    #         inputs = tf.nn.embedding_lookup(self.embedding, self.x_new_seq)
    #         self.input_t_seq = tf.unstack(inputs, self.sample_updates, 0)
    #     else:
    #         self.input_t_seq = tf.unstack(self.x_new_seq, self.sample_updates, axis=0)
    #     if self.use_lstm:
    #         l = tf.unstack(state, axis=0)
    #         state_lstm = tf.nn.rnn_cell.LSTMStateTuple(l[0],l[1])
    #         self.lstm_outputs_t_seq,self.state_t_seq=tf.nn.static_rnn(self.cell,self.input_t_seq,dtype="float32",initial_state=state_lstm)
    #     else:
    #         self.lstm_outputs_t_seq, self.state_t_seq = tf.nn.static_rnn(self.cell, self.input_t_seq,
    #                                                                      dtype="float32", initial_state=state)
    #
    #     if self.dataset == 'ptb':
    #         self.lstm_outputs_t_seq = tf.reshape(self.lstm_outputs_t_seq, [-1, self.num_units])
    #         self.output_t_logits_seq = tf.nn.xw_plus_b(self.lstm_outputs_t_seq, self.W, self.b)
    #         self.output_t_logits_seq = tf.reshape(self.output_t_logits_seq, [self.batch_size, self.sample_updates, self.vocab_size])
    #         target = tf.reshape(self.y_t[-1],[self.batch_size,1])
    #         loss = tf.contrib.seq2seq.sequence_loss(
    #                     tf.reshape(self.output_t_logits_seq[:,-1,:],[self.batch_size,1,tf.shape(self.output_t_logits_seq)[-1]]),
    #                     target,
    #                     tf.ones([self.batch_size, 1], dtype=tf.float32),
    #                     average_across_timesteps=False,
    #                     average_across_batch=True)
    #         # Update the cost
    #         self.loss_target_seq = tf.reduce_sum(loss)
    #         self.loss_state_seq = self.lambda_state*tf.losses.mean_squared_error(state_t,self.state_t_seq)
    #
    #     else:
    #         self.output_t_logits_seq = tf.matmul(self.lstm_outputs_t_seq[-1], self.W) + self.b
    #         self.loss_state_seq = self.lambda_state*tf.losses.mean_squared_error(state_t,self.state_t_seq)
    #
    #         self.y_as_list_seq = tf.unstack(self.y_t, num=self.sample_updates, axis=0)
    #         self.loss_target_seq = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y_t[-1], self.output_t_logits_seq))
    #
    #     if self.use_buffer_bptt is True:
    #         self.total_loss_seq = tf.reduce_mean(self.loss_target_seq)
    #     else:
    #         self.total_loss_seq = tf.reduce_mean(tf.add(self.loss_target_seq, self.loss_state_seq))
    #         grad = tf.gradients(self.total_loss_seq,state)
    #         state -= tf.squeeze(tf.multiply(self.state_learning_rate,grad), axis=0)
    #         grad = tf.gradients(self.total_loss_seq,state_t)
    #         state_t -= tf.squeeze(tf.multiply(self.state_learning_rate,grad), axis=0)
    #
    #     self.state_tm1_c_seq = state
    #     self.state_t_c_seq = state_t
    #
    # def create_train_sequential(self):
    #     lstm_variables = [v for v in tf.trainable_variables() if v.name.startswith('rnn')]
    #     with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    #         optimizer_lstm = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    #         gvs_lstm = optimizer_lstm.compute_gradients(self.total_loss_seq, var_list=lstm_variables)
    #         if self.clip_gradients:
    #             capped_gvs_lstm = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs_lstm]
    #             self.train_lstm = optimizer_lstm.apply_gradients(capped_gvs_lstm)
    #         else:
    #             self.train_lstm = optimizer_lstm.apply_gradients(gvs_lstm)
    #
    #         optimizer_output = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_output)
    #         gvs_output = optimizer_output.compute_gradients(self.total_loss_seq, var_list = [self.W,self.b])
    #         if self.clip_gradients:
    #             capped_gvs_output = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs_output]
    #             self.train_output = optimizer_output.apply_gradients(capped_gvs_output)
    #         else:
    #             self.train_output = optimizer_output.apply_gradients(gvs_output)
    #
    #     self.train_seq = tf.group(self.train_lstm, self.train_output)