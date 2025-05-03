import json
import nltk
import math
import tqdm
import numpy as np
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from collections import Counter, defaultdict

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')


class ParagraphProcessor:
    def __init__(self, words_path: str|None, lang='english', removed=[], unkonwn='<UNK>'):
        self.lemmatizer = WordNetLemmatizer()
        self.UNK_TOKEN = unkonwn
        self.STOPWORDS = set(stopwords.words(lang))
        self.REMOVED_WORDS = removed
        self.labels = {
            'cultural exclusive': 0,
            'cultural agnostic': 1,
            'cultural representative': 2
        }
        for word in removed:
            self.STOPWORDS.add(word)
        self.ENGLISH_WORDS = set(words.words())
        with open(f'{words_path}', 'r') as file:
            self.features = json.load(file)
            file.close()
        self.penalizedWords = self.get_penal_words(self.features)     
    
    def get_penal_words(self, features):
        penalization_words = []
        suspicious_words = defaultdict(list)

        for label, word_weights in features.items():
            for word, weight in word_weights:
                if (
                    word.lower() not in self.ENGLISH_WORDS and 
                    word.lower() not in self.STOPWORDS
                ):
                    suspicious_words[label].append((word, weight))

        for label, words in suspicious_words.items():
            for word, score in words:
                penalization_words.append(word)
        return penalization_words
    
    def tokenize(self, doc):
        tokens = word_tokenize(doc.lower())
        tokens = [w for w in tokens if w.isalpha() and w not in self.STOPWORDS]
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        return tokens
    
    def tf(self, doc):
        tf_ = {}
        words = self.tokenize(doc)
        cnt = Counter(words)
        normalization_factor = len(words)
        for word, occs in cnt.items():
            tf_[word] = occs / normalization_factor
        return tf_
    
    def idf(self, texts):
        idf_ = {}
        N = len(texts)
        tokenized_docs = [set(self.tokenize(doc)) for doc in texts]
        all_words = set(word for doc in tokenized_docs for word in doc)
        
        for word in all_words:
            doc_cnt = sum(1 for doc in tokenized_docs if word in doc)
            idf_[word] = math.log((1 + N) / (1 + doc_cnt)) + 1
        return idf_

    def idf_by_category(self, df):
        categorical_idf = {}
        grouped = df.groupby('category')['paragraph'].apply(list)

        for cat, docs in grouped.items():
            categorical_idf[cat] = self.idf(docs)

        return categorical_idf
    
    def tf_idf(self, tf_, idf_, word_index=None):
        unk_token = self.UNK_TOKEN
        tfidf = {}
        unk_val = 0.0

        for word, tf_val in tf_.items():
            idf_val = idf_.get(word, 1.0)
            penalty = 0.3 if word in self.penalizedWords else 1.0
            value = tf_val * idf_val * penalty

            if word_index is not None and word not in word_index:
                unk_val += value
            else:
                tfidf[word] = value

        if word_index is not None and unk_token in word_index:
            tfidf[unk_token] = unk_val

        return tfidf

    def vectorize(self, tfidf_dict, word_index):
        vec = np.zeros(len(word_index))
        unk_idx = word_index.get(self.UNK_TOKEN)
        
        for word, value in tfidf_dict.items():
            if word in word_index:
                vec[word_index[word]] = value
            elif unk_idx is not None:
                vec[unk_idx] += value
        return vec
    
    def process(self, testDataset, validDataset, trainDataset):
        idf_cat = self.idf_by_category(trainDataset)
        train_tfidfs = []
        test_tfidfs = []
        val_tfidfs = []
        
        for _, row in tqdm.tqdm(trainDataset.iterrows(), total=len(trainDataset), colour='green'):
            doc = row['paragraph']
            cat = row['category']
            tf_ = self.tf(doc)
            idf_ = idf_cat.get(cat, {})
            tfidf = self.tf_idf(tf_, idf_)
            train_tfidfs.append(tfidf)

        trainDataset['tf_idf'] = train_tfidfs
        
        vocab = sorted(set(word for doc in trainDataset['paragraph'] for word in self.tokenize(doc)))
        word_index = {word: idx for idx, word in enumerate(vocab)}
        word_index[self.UNK_TOKEN] = len(word_index)
        
        for _, row in tqdm.tqdm(testDataset.iterrows(), total=len(testDataset), colour='green'):
            doc = row['paragraph']
            cat = row['category']
            tf_ = self.tf(doc)
            idf_ = idf_cat.get(cat, {})
            tfidf = self.tf_idf(tf_, idf_, word_index=word_index)
            test_tfidfs.append(tfidf)
        testDataset['tf_idf'] = test_tfidfs

        for _, row in tqdm.tqdm(validDataset.iterrows(), total=len(validDataset), colour='green'):
            doc = row['paragraph']
            cat = row['category']
            tf_ = self.tf(doc)
            idf_ = idf_cat.get(cat, {})
            tfidf = self.tf_idf(tf_, idf_, word_index=word_index)
            val_tfidfs.append(tfidf)
        validDataset['tf_idf'] = val_tfidfs

        test_docs = testDataset['tf_idf'].to_list()
        
        dictionary = trainDataset['tf_idf'].to_list()
        
        val_docs = validDataset['tf_idf'].to_list()
        
        XTrain = np.array([self.vectorize(doc_tfidf, word_index) for doc_tfidf in dictionary])
        XVal = np.array([self.vectorize(doc_tfidf, word_index) for doc_tfidf in val_docs])
        XTest = np.array([self.vectorize(doc_tfidf, word_index) for doc_tfidf in test_docs])
        return XTest, XTrain, XVal