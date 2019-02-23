import numpy as np
import pickle, os, random

tag2label = {
    'O': 0,
    'B-PER':1, 'I-PER':2,
    'B-LOC':3, 'I-LOC':4,
    'B-ORG':5, 'I-ORG':6
}


def get_train_data_len(corpus_path):
    # line_num = 0
    # with open(corpus_path, encoding='utf-8') as fr:
    #     line = fr.readline()
    #     while line:
    #         if line == '\n':
    #             line_num += 1
    #             if line_num % 1000000 == 0:
    #                 print(line_num)
    #         line = fr.readline()
    line_num = 16823089
    print(line_num)
    return line_num


def read_corpus(corpus_path):
    '''
    从训练文件读取数据
    中   B_LOC
    国   I_LOC
    很   O
    大   O

    句子与句子之间用空行隔开

    输出格式：
    data = [(['中', '国', '很', '大'], [B_LOC, I_LOC, O, O]), ...]
    '''
    with open(corpus_path, encoding='utf-8') as fr:
        line = fr.readline()
        sent_, tag_ = [], []
        while line:
            if line != '\n':
                sent, tag = line.strip().split()
                sent_.append(sent)
                tag_.append(tag)
            elif sent_:
                yield sent_, tag_
                sent_, tag_ = [], []

            line = fr.readline()


def vocab_build(vocab_path, corpus_path, min_count):
    '''
    建立字典
    word2id = {word : [id, cnt]}
    根据word的cnt去掉出现次数小于min_count的词
    id从 1 开始

    word2id = {word : id}
    最后pickle序列化保存word2id
    '''
    word2id = {}
    data = read_corpus(corpus_path)
    for sent, _ in data:
        for word in sent:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'

            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1

    # 去掉次数小于min_count的词
    low_freq_words = []
    for word, [_, cnt] in word2id.items():
        if word != '<ENG>' and word != '<NUM>' and cnt < min_count:
            low_freq_words.append(word)

    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


# vocab_build('./word2id/word2id.pkl', './train_data/train_data', 100)

def read_dictionary(vocab_path):
    '''
    读取pickle序列化的word2id
    '''
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab size is:', len(word2id))
    return word2id


def sentence2id(sent, word2id):
    '''
    查到句子中的各个词的id返回
    '''
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        elif word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])

    return sentence_id


def random_embedding(word2id, embedding_dim):
    '''
    获得各 word 的随机vector
    '''
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(word2id), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def batch_yield(corpus_path, batch_size, word2id, tag2label):
    '''
    word -> id
    tag -> label
    产生batch_size句话的 id 和 label
        [[id1, ..., idn], ...]
        [[label1, ..., labeln], ...]
    '''
    seqs, labels = [], []
    data = read_corpus(corpus_path)
    for sent_, tag_ in data:
        sent_ = sentence2id(sent_, word2id)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if seqs:
        yield seqs, labels


def pad_sequences(sequences, pad_mark=0):
    '''
    将不等长序列pad成一样长度
    返回pad后的序列，和序列pad之前的长度
    '''
    seq_list, seq_len_list = [], []
    max_len = max(map(lambda x : len(x), sequences))
    for seq in sequences:
        seq = list(seq)
        seq_len_list.append(len(seq))

        seq = seq[:max_len] + [pad_mark] * max(0, max_len - len(seq))
        seq_list.append(seq)

    return seq_list, seq_len_list

# def batch_yield(data, batch_size, word2id, tag2label, shuffle=False):
#     '''
#     word -> id
#     tag -> label
#     产生batch_size句话的 id 和 label
#         [[id1, ..., idn], ...]
#         [[label1, ..., labeln], ...]
#     '''
#     if shuffle:
#         random.shuffle(data)
#
#     seqs, labels = [], []
#     for sent_, tag_ in data:
#         sent_ = sentence2id(sent_, word2id)
#         label_ = [tag2label[tag] for tag in tag_]
#
#         if len(seqs) == batch_size:
#             yield seqs, labels
#             seqs, labels = [], []
#
#         seqs.append(sent_)
#         labels.append(label_)
#
#     if seqs:
#         yield seqs, labels













