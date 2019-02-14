from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import tensorflow as tf
from data import pad_sequences

class BiLSTM_CRF(object):
    def __init__(self, embedding, hidden_dim):
        self.embedding = embedding
        self.hidden_dim = hidden_dim


    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()      # 全连接层，通过 word_id 去寻找 word vector
        self.biLSTM_layer_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, [None, None], name='word_ids')
        self.labels = tf.placeholder(tf.int32, [None, None], name='labels')
        self.sequence_lengths = tf.placeholder(tf.int32, [None], name='sequence_lengths')
        self.dropout_pl = tf.placeholder(tf.float32, [None], name='dropout')

    def lookup_layer_op(self):
        with tf.variable_scope('words'):
            _word_embeddings = tf.Variable(self.embedding,
                                           dtype=tf.float32,
                                           name='_word_embeddings')
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                     self.word_ids,
                                                     name='word_embeddings')
            self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        with tf.variable_scope('bi-lstm'):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = cell_fw,
                cell_bw = cell_bw,
                inputs = self.word_embeddings,
                sequence_length = self.sequence_lengths,
                dtype = tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope('probj'):
            W = tf.get_variable('W',
                                shape=[2*self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable('b',
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])


    def loss_op(self):
        if self.CRF:







    def get_feed_dict(self, seqs, labels=None, dropout=None):
        word_ids, seq_len_list = pad_sequences(seqs)
        feed_dict = {self.word_ids: word_ids, self.sequence_lengths: seq_len_list}

        if labels is not None:
            labels, _ = pad_sequences(labels)
            feed_dict[self.labels] = labels

        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict


