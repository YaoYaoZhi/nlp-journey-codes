import itertools
import logging
import os
import pickle
import time

from tensorflow import keras

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

from nlp.utils import *

log = Log(logging.INFO)


class SiameseSimilarity:
    def __init__(self, model_path,
                 config_path,
                 data_path=None,
                 embedding_file=None,
                 n_hidden=128,
                 batch_size=64,
                 epochs=10,
                 embedding_dim=300,
                 train=False):
        """
        初始化
        :param model_path: 要保存的或者已经保存的模型路径
        :param config_path: 要保存的或者已经保存的配置文件路径
        :param data_path: 存放了train.csv和test.csv的目录
        :param embedding_file: 训练好的词向量文件
        :param n_hidden: lstm隐藏层维度
        :param batch_size: 每批数目大小
        :param epochs: 
        :param train: 是否训练模式，如果是训练模式，则必须提供data_path
        """

        self.model_path = model_path
        self.config_path = config_path
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden

        # 加载停用词
        self.stops = set_en_stopwords()
        self.embeddings, self.word_index, self.max_length = self._load_config()
        if not train:
            self.model = self._load_model()
        else:
            assert data_path is not None, '训练模式，训练数据必须！'
            assert embedding_file is not None, '训练模式，训练好的词向量数据必须！'
            self.data_path = data_path
            self.batch_size = batch_size
            self.epochs = epochs
            self.embedding_file = embedding_file
            if self.embeddings is not None:
                self.x_train, self.y_train, self.x_val, self.y_val, _, _ = self._load_data()
            else:
                self.x_train, self.y_train, self.x_val, self.y_val, self.word_index, self.max_length = self._load_data()
                self.embeddings = self._load_word2vec(self.word_index)
            self.model = self.train(call_back=True)

    def __build_model(self):
        log.info('构建模型...')
        left_input = keras.layers.Input(shape=(self.max_length,), dtype='int32')
        right_input = keras.layers.Input(shape=(self.max_length,), dtype='int32')
        embedding_layer = keras.layers.Embedding(len(self.embeddings),
                                                 self.embedding_dim,
                                                 weights=[self.embeddings],
                                                 input_length=self.max_length,
                                                 trainable=False)
        # Embedding
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)
        # 相同的lstm网络
        shared_lstm1 = keras.layers.Bidirectional(keras.layers.LSTM(self.n_hidden // 2, return_sequences=True))
        shared_lstm2 = keras.layers.Bidirectional(keras.layers.LSTM(self.n_hidden // 2, return_sequences=True))
        shared_lstm3 = keras.layers.Bidirectional(keras.layers.LSTM(self.n_hidden // 2, return_sequences=True))
        shared_lstm4 = keras.layers.Bidirectional(keras.layers.LSTM(self.n_hidden // 2))

        left_output = shared_lstm1(encoded_left)
        left_output = shared_lstm2(left_output)
        left_output = shared_lstm3(left_output)
        left_output = shared_lstm4(left_output)

        right_output = shared_lstm1(encoded_right)
        right_output = shared_lstm2(right_output)
        right_output = shared_lstm3(right_output)
        right_output = shared_lstm4(right_output)

        # 合并后计算
        merged = keras.layers.concatenate([left_output, right_output])
        merged = keras.layers.BatchNormalization()(merged)
        merged = keras.layers.Dropout(0.5)(merged)
        merged = keras.layers.Dense(32, activation='relu')(merged)
        merged = keras.layers.BatchNormalization()(merged)
        merged = keras.layers.Dropout(0.5)(merged)
        output = keras.layers.Dense(1, activation='sigmoid')(merged)
        # 构造模型
        model = keras.models.Model([left_input, right_input], [output])
        # Adam 优化器
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model

    def train(self, weights_only=True, call_back=False):
        model = self.__build_model()

        if call_back:
            callbacks = self.build_callbacks(weights_only)
        else:
            callbacks = None
        log.info('开始训练...')
        model_trained = model.fit([self.x_train['left'], self.x_train['right']],
                                  self.y_train,
                                  batch_size=self.batch_size,
                                  epochs=self.epochs,
                                  validation_data=([self.x_val['left'], self.x_val['right']], self.y_val),
                                  verbose=1,
                                  callbacks=callbacks)
        log.info('训练完毕...')
        if weights_only and not call_back:
            model.save_weights(os.path.join(
                self.model_path, 'weights_only.h5'))
        elif not weights_only and not call_back:
            model.save(os.path.join(self.model_path, 'model.h5'))
        self._save_config()
        plot(model_trained)
        return model

    def build_callbacks(self, weights_only):
        log.info('构建callbacks...')
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
        stamp = 'lstm_%d' % self.n_hidden
        checkpoint_dir = os.path.join(
            self.model_path, 'checkpoints/' + str(int(time.time())) + '/')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + stamp + '.h5'
        if weights_only:
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                bst_model_path, save_best_only=True, save_weights_only=True)
        else:
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                bst_model_path, save_best_only=True)
        tensor_board = keras.callbacks.TensorBoard(
            log_dir=checkpoint_dir + "logs/{}".format(time.time()))
        callbacks = [early_stopping, model_checkpoint, tensor_board]
        return callbacks

    def _save_config(self):
        with open(self.config_path, 'wb') as out:
            pickle.dump((self.embeddings, self.word_index, self.max_length), out)
        if out:
            out.close()

    # 推理两个文本的相似度，大于0.5则相似，否则不相似
    def predict(self, text1, text2):
        if isinstance(text1, list) or isinstance(text2, list):
            x1 = [[self.word_index.get(word, 0) for word in clean_to_list(
                text)] for text in text1]
            x2 = [[self.word_index.get(word, 0) for word in clean_to_list(
                text)] for text in text2]
            x1 = keras.preprocessing.sequence.pad_sequences(x1, maxlen=self.max_length)
            x2 = keras.preprocessing.sequence.pad_sequences(x2, maxlen=self.max_length)
        else:
            x1 = [self.word_index.get(word, 0)
                  for word in clean_to_list(text1)]
            x2 = [self.word_index.get(word, 0)
                  for word in clean_to_list(text2)]
            x1 = keras.preprocessing.sequence.pad_sequences([x1], maxlen=self.max_length)
            x2 = keras.preprocessing.sequence.pad_sequences([x2], maxlen=self.max_length)
        # 转为词向量
        return self.model.predict([x1, x2])

    # 保存路径与加载路径相同
    def _load_model(self, weights_only=True):
        return self._load_model_by_path(self.model_path, weights_only)

    # 自定义加载的模型路径
    def _load_model_by_path(self, model_path, weights_only=True):
        try:
            if weights_only:
                model = self.__build_model()
                model.load_weights(model_path)
            else:
                model = keras.models.load_model(model_path)
        except FileNotFoundError:
            model = None
        return model

    # 加载word2vec词向量
    def _load_word2vec(self, word_index):
        log.info('加载训练好的词向量')
        word2vec = KeyedVectors.load_word2vec_format(
            self.embedding_file, binary=True)
        embeddings = 1 * np.random.randn(len(word_index) + 1, self.embedding_dim)
        embeddings[0] = 0

        for word, index in word_index.items():
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)
        return embeddings

    def _load_config(self):
        log.info('加载配置文件（词向量和最大长度）')
        try:
            with open(self.config_path, 'rb') as config:
                embeddings, vocabulary, max_seq_length = pickle.load(config)
            if config:
                config.close()
        except FileNotFoundError:
            embeddings, vocabulary, max_seq_length = None, None, None
        return embeddings, vocabulary, max_seq_length

    def _load_data(self, test_size=0.2):
        # word:index和index:word
        word_index = dict()
        index_word = ['<unk>']
        questions_cols = ['question1', 'question2']

        log.info('加载数据集...')
        train_data = os.path.join(self.data_path, 'train.csv')
        test_data = os.path.join(self.data_path, 'test.csv')

        train_df = pd.read_csv(train_data).head(100)
        test_df = pd.read_csv(test_data).head(100)

        # 找到最大的句子长度
        sentences = [df[col].str.split(' ') for df in [train_df, test_df] for col in questions_cols]
        max_length = max([len(s) for ss in sentences for s in ss if isinstance(s, list)])
        log.info('处理数据集...')
        # 预处理(统计并将字符串转换为索引)
        for dataset in [train_df, test_df]:
            for index, row in dataset.iterrows():
                for question_col in questions_cols:
                    question_indexes = []
                    for word in clean_to_list(row[question_col]):
                        if word in self.stops:
                            continue
                        if word not in word_index:
                            word_index[word] = len(index_word)
                            question_indexes.append(len(index_word))
                            index_word.append(word)
                        else:
                            question_indexes.append(word_index[word])
                    # dataset.set_value(index, question_col, question_indexes) # 已过期的函数
                    dataset.at[index, question_col] = question_indexes

        log.info('数据集处理完毕...')

        x = train_df[questions_cols]
        y = train_df['is_duplicate']
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size)

        x_train = {'left': x_train.question1, 'right': x_train.question2}
        x_val = {'left': x_val.question1, 'right': x_val.question2}

        y_train = y_train.values
        y_val = y_val.values

        for dataset, side in itertools.product([x_train, x_val], ['left', 'right']):
            dataset[side] = keras.preprocessing.sequence.pad_sequences(dataset[side], maxlen=max_length)

        # 校验问题对各自数目是否正确
        assert x_train['left'].shape == x_train['right'].shape
        assert len(x_train['left']) == len(y_train)
        return x_train, y_train, x_val, y_val, word_index, max_length
