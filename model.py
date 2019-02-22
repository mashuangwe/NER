import time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from data import pad_sequences, batch_yield
from utils import get_logger


class BiLSTM_CRF(object):
    def __init__(self, FLAGS, embeddings, server,
                 word2id, num_workers,
                 tag2label, paths, train_data_len):
        self.FLAGS = FLAGS
        self.shuffle = FLAGS.shuffle
        self.clip_grad = FLAGS.clip
        self.lr = FLAGS.lr
        self.server = server
        self.epoch_num = FLAGS.epoch
        self.num_workers = num_workers
        self.embeddings = embeddings
        self.embedding_update = FLAGS.update_embedding 
        self.hidden_dim = FLAGS.hidden_dim
        self.optimizer = FLAGS.optimizer
        self.batch_size = FLAGS.batch_size
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.batch_size = FLAGS.batch_size
        self.word2id = word2id
        self.dropout_keep_prob = FLAGS.dropout
        # self.train_data = train_data
        self.train_data_len = train_data_len

        self.train_data_source = paths['train_data_source']
        self.train_path = paths['train_path']
        self.log_path = paths['log_path']
        self.logger = get_logger(paths['log_file'])
        self.model_file_prefix = paths['model_file_prefix']


    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.loss_op()
        self.train()


    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.dropout_pl = tf.placeholder(tf.float32, shape=[], name='dropout')


    def lookup_layer_op(self):
        with tf.variable_scope('words'):
            _word_embeddings = tf.Variable(initial_value=self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.embedding_update,
                                           name='_word_embeddings')
            self.word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                          ids=self.word_ids,
                                                          name='word_embeddings')
        self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout_pl)


    def biLSTM_layer_op(self):
        with tf.variable_scope('bi-lstm'):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)

            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                sequence_length=self.sequence_lengths,
                inputs=self.word_embeddings,
                dtype=tf.float32)
            '''
            output_fw_seq的shape [batch_size, max_time, cell_fw.output_size]
            output_bw_seq的shape [batch_size, max_time, cell_bw.output_size]
            '''
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)     # 全连接层之前先要 dropout

        with tf.variable_scope('proj'):
            W = tf.get_variable('W',
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable('b',
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])


    def loss_op(self):
        log_likelihood, self.transition_params = crf_log_likelihood(
            inputs=self.logits,
            tag_indices=self.labels,
            sequence_lengths=self.sequence_lengths)
        self.loss = -tf.reduce_mean(log_likelihood)


    def train(self):
        with tf.variable_scope('train_step'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradDAOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

            # distributed optm
            if self.FLAGS.sync_replicas:
                if self.FLAGS.replicas_to_aggregate is None:
                    self.replicas_to_aggregate = self.num_workers
                else:
                    self.replicas_to_aggregate = self.FLAGS.replicas_to_aggregate

            optim = tf.train.SyncReplicasOptimizer(
                opt=optim,
                replicas_to_aggregate=self.replicas_to_aggregate,
                total_num_replicas=self.num_workers,
                name='bilstm_crf_sync_replicas')

            grad_and_vars = optim.compute_gradients(self.loss)
            grad_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grad_and_vars]
            self.train_op = optim.apply_gradients(grad_and_vars_clip, global_step=self.global_step)

            self.is_chief = self.FLAGS.task_index == 0
            if self.FLAGS.sync_replicas:
                local_init_op = optim.local_step_init_op
                if self.is_chief:
                    local_init_op = optim.chief_init_op

                ready_for_local_init_op = optim.ready_for_local_init_op

                # Initial token and chief queue runners required by the sync_replicas mode
                chief_queue_runner = optim.get_chief_queue_runner()
                sync_init_op = optim.get_init_tokens_op()

            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables())

            if self.FLAGS.sync_replicas:
                sv = tf.train.Supervisor(
                    is_chief=self.is_chief,
                    ready_for_local_init_op=ready_for_local_init_op,
                    init_op=self.init_op,
                    local_init_op=local_init_op,
                    logdir=self.log_path,
                    saver=self.saver,
                    global_step=self.global_step,
                    recovery_wait_secs=1)
            else:
                sv = tf.train.Supervisor(
                    is_chief=self.is_chief,
                    init_op=self.init_op,
                    logdir=self.log_path,
                    global_step=self.global_step,
                    recovery_wait_secs=1)

            gpu_options = tf.GPUOptions(
                allow_growth = True,
                allocator_type = 'BFC',
                visible_device_list = '%d' % self.FLAGS.task_index)

            sess_config = tf.ConfigProto(
                gpu_options=gpu_options,
                allow_soft_placement=True,
                log_device_placement=False,
                device_filters=['/job:ps', '/job:worker/task:%d' % self.FLAGS.task_index])
            self.config = sess_config

            # The chief worker (task_index == 0) session will prepare the session,
            # while the remaining workers will wait for the preparation to complete.
            if self.is_chief:
                print('Worker: %d: Initializing session ...' % self.FLAGS.task_index)
            else:
                print('Worker: %d: Waiting for session to be initialized ...' % self.FLAGS.task_index)

            worker_spec = self.FLAGS.worker_hosts.split(',')
            if self.FLAGS.existing_servers:
                server_grpc_url = 'grpc://' + worker_spec[self.FLAGS.task_index]
                print('using existing server at: %s', server_grpc_url)
                sess = sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config)
            else:
                sess = sv.prepare_or_wait_for_session(self.server.target, config=sess_config)

            print('Worker %d: Session initialization complete.' % self.FLAGS.task_index)
            if self.FLAGS.sync_replicas and self.is_chief:
                # chief worker will start the queue runner and call the init op
                sess.run(sync_init_op)
                sv.start_queue_runners(sess, [chief_queue_runner])

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, self.train_data_source, self.train_data_len, self.tag2label, epoch)


    def run_one_epoch(self, sess, train_data_source, train_data_len, tag2label, epoch):
        num_batches = (train_data_len + self.batch_size - 1) // self.batch_size

        starttime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        # batches = batch_yield(train, self.batch_size, self.word2id, self.tag2label, shuffle=self.shuffle)
        batches = batch_yield(train_data_source, self.batch_size, self.word2id, self.tag2label)

        for step, (seqs, labels) in enumerate(batches):
            sys.stdout.write('processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
			
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.dropout_keep_prob)
            _, loss_train, _ = sess.run(
                [self.train_op, self.loss, self.global_step], feed_dict=feed_dict)

            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(
                        starttime, epoch + 1, step + 1, loss_train, step_num))

            if step + 1 == num_batches:
                self.saver.save(sess, self.model_file_prefix, global_step=step_num)


    def get_feed_dict(self, seqs, labels=None, dropout=None):
        '''
        返回各句子的各 word 的 id 以及句子长度， 甚至各句子的各 word 的 label
        '''
        word_ids, seq_len_lists = pad_sequences(seqs, pad_mark=0)
        feed_dict = {self.word_ids: word_ids, self.sequence_lengths: seq_len_lists}

        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_

        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_lists












