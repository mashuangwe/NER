import tensorflow as tf
import numpy as np
import os, time
from model import BiLSTM_CRF
from utils import get_logger
from data import read_corpus, read_dictionary, tag2label, random_embedding

flags = tf.app.flags
flags.DEFINE_string('mode', 'train', 'train/test/demo')
flags.DEFINE_string('train_data_path', 'train_data', 'train data path')
flags.DEFINE_string('test_data_path', 'test_data', 'test data path')
flags.DEFINE_string('word2id', 'word2id', 'word vector path')
flags.DEFINE_string('test_data', 'test_data', 'test data source')
flags.DEFINE_integer('batch_size', 128, 'sample of each minibatch')
flags.DEFINE_integer('epoch', 2, 'epoch of training')
flags.DEFINE_integer('hidden_dim', 100, 'dim of hidden state')
flags.DEFINE_string('optimizer', 'Adam', 'Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
flags.DEFINE_boolean('CRF', True, 'use CRF at the top layer. If False, use Softmax')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_float('clip', 5.0, 'gradient clipping')
flags.DEFINE_float('dropout', 0.5, 'dropout keep_prob')
flags.DEFINE_boolean('update_embedding', True, 'update embedding during training')
flags.DEFINE_string('pretrain_embedding', 'random', 'use pretraind word embedding or init it randomly')
flags.DEFINE_integer('embedding_dim', 100, 'random init word embedding dim')
flags.DEFINE_boolean('shuffle', True, 'shuffle training data before each epoch')

flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 4, "Total number of gpus for each machine."
                                    "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update "
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_boolean("sync_replicas", True,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_boolean("existing_servers", False, "Whether servers already exists. If True, "
                     "will use the worker hosts via their GRPC URLs (one client process "
                     "per worker host). Otherwise, will create an in-process TensorFlow "
                     "server.")

flags.DEFINE_string("ps_hosts", "172.16.23.5:2225", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts",
                    "172.16.23.5:2226,172.16.23.5:2228",
                    "Comma-separated list of hostname:port pairs")

# flags.DEFINE_string("worker_hosts",
#                     "172.16.23.5:2223,172.16.23.5:2224,172.16.23.5:2225,172.16.23.5:2226,"
#                     "172.16.23.11:2223,172.16.23.11:2224,172.16.23.11:2225,172.16.23.11:2226",
#                     "Comma-separated list of hostname:port pairs")

flags.DEFINE_string("job_name", None, "job name: worker or ps")
FLAGS = flags.FLAGS

# get word embeddings
word2id = read_dictionary(os.path.join('./', FLAGS.word2id, 'word2id.pkl'))
if FLAGS.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, FLAGS.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')

# read corpus and get training data
if FLAGS.mode != 'demo':
    train_path = os.path.join('.', FLAGS.train_data_path, 'train_data')
    # test_path = os.path.join('.', FLAGS.test_data_path, 'test_data')
    train_data = read_corpus(train_path)
    # test_data = read_corpus(test_path)


# path setting
paths = {}

# output_path
output_path = os.path.join('./', 'output', time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
if not os.path.exists(output_path):
    os.makedirs(output_path)

# model_path
model_path = os.path.join('./', output_path, 'model')
if not os.path.exists(model_path):
    os.makedirs(model_path)
paths['model_file_prefix'] = os.path.join(model_path, 'model')

# log_path
log_path = os.path.join('./', output_path, 'log')
paths['log_path'] = log_path
if not os.path.exists(log_path):
    os.makedirs(log_path)

log_file = os.path.join(log_path, 'log.txt')
paths['log_file'] = log_file
get_logger(log_file).info(str(FLAGS))

# train path
train_path = os.path.join('./', output_path, 'train')
if not os.path.exists(train_path):
    os.makedirs(train_path)
paths['train_path'] = train_path

if FLAGS.mode == 'train':
    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('must specify an explicit `job_name`')
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('must specify an explicit `task_index`')

    print('job_name = %s' % FLAGS.job_name)
    print('task_index = %s' % FLAGS.task_index)

    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')
    num_workers = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})

    if not FLAGS.existing_servers:
        # not using existing servers, Create an in-process server
        server = tf.train.Server(
            cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        if FLAGS.job_name == 'ps':
            server.join()

    if FLAGS.num_gpus > 0:
        # Avoid gpu allocation conflict: now allocate task_num -> #gpu
        # for each worker in the corresponding machine
        # gpu = FLAGS.task_index % FLAGS.num_gpus
        # print('gpu ====== %d' % gpu)
        gpu = 0
        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
    elif FLAGS.num_gpus == 0:
        # Just allocate the CPU to worker server
        cpu = 0
        worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
        # The device setter will automatically place Variables ops on separate
        # parameter servers (ps). The non-Variable ops will be placed on the workers.
        # The ps use CPU and workers use corresponding GPU

    with tf.device(tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_device='/job:ps/cpu:0',
        cluster=cluster)):
        model = BiLSTM_CRF(FLAGS=FLAGS,
                           embeddings=embeddings,
                           server=server,
                           num_workers=num_workers,
                           word2id=word2id,
                           tag2label=tag2label,
                           paths=paths,
                           train_data=train_data)
        model.build_graph()












