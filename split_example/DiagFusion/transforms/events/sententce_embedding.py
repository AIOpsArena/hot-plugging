import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from DiagFusion.public_function import load, save
import numpy as np


# tfidf * word embedding


def read_text(path):
    text = []
    f = open(path, 'r')
    line = f.readline()
    #     text.append(line[:-12])
    text.append(line.split('\t')[0])
    while line:
        line = f.readline()
        #         text.append(line[:-12])
        text.append(line.split('\t')[0])
    f.close()
    # 去最后的空串
    return text[:-1]


def sentence_embedding(file_dict, train_path, save_dir, save_path, service_num):
    data_dict = load(file_dict)

    train_text = read_text(train_path)
    vectorizer = CountVectorizer(lowercase=False,
                                 token_pattern=r'(?u)\b\S\S+')  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    vec_train = vectorizer.fit_transform(train_text)
    tfidf_train = transformer.fit_transform(vec_train)

    weight_train = tfidf_train.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

    word = vectorizer.get_feature_names_out()  # 获取词袋模型中的所有词语
    word_dict = {word[i]: i for i in range(len(word))}

    train_embedding = tfidf_word_embedding(weight_train, data_dict, train_text, word_dict, service_num)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('sentence_embedding shape:',
          f'{len(train_embedding)} * {len(train_embedding[0])} * {len(train_embedding[0][0])}')
    save(save_path, train_embedding)


def tfidf_word_embedding(weight, data_dict, texts, word_dict, service_num):
    length = len(data_dict[list(data_dict.keys())[0]])
    count = 0
    case_embedding = []
    sentence_embedding = []
    for text in texts:
        temp = np.array([0] * length, 'float32')
        if text != '':
            words = list(set(text.split(' ')))
            for word in words:
                try:
                    if word in word_dict:
                        temp = temp + weight[count][word_dict[word]] * np.array(data_dict[word])
                except:
                    continue
        case_embedding.append(temp)
        if (count + 1) % service_num == 0:
            sentence_embedding.append(case_embedding)
            case_embedding = []
        count += 1
    return sentence_embedding


def run_sentence_embedding(config):
    sentence_embedding(config['source_path'], config['train_path'],
                       config['save_dir'], config['save_path'], config['K_S'])