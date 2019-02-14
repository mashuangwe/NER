import pickle, os, random
import numpy as np

# tags, BIO
tag2label = {
    'O': 0,
    'B-LOC': 1, 'I-LOC': 2,
    'B-PER': 3, 'I-PER': 4,
    'B-ORG': 5, 'I-ORG': 6,
}

def read_corpus(corpus_path):
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
        sent_, tag_ = [], []
        for line in lines:
            if line != '\n':
                char, label = line.strip().split()
                sent_.append(char)
                tag_.append(label)
            else:
                data.append((sent_, tag_))
                sent_, tag_ = [], []

    return data

def vocab_build(vocab_path, corpus_path, min_count):
    data = read_corpus(corpus_path)
    word2id = {}

    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'

            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1

    low_freq_words = []
    for word, [_, word_freq] in word2id.items():
        if word_freq < min_count and word != '<ENG>' and word != '<NUM>':
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

def sentence2id(sent, word2id):
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

def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size: ', len(word2id))
    return word2id

def batch_yield(data, batch_size, word2id, tag2label, shuffle=False):
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for sent_, tag_ in data:
        sent_ = sentence2id(sent_, word2id)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

def pad_sequences(sequences, pad_mark=0):
    '''
    将各sequence补0成一样长
    '''
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max((max_len - len(seq), 0))
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))

    return seq_list, seq_len_list

def random_embeddings(word2id, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(word2id), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat













