import pandas as pd
pd.set_option('display.max_rows', 1000)
import numpy as np
import string
import transformers
from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification, AdamW

df = pd.read_csv("../../code/conll03/data/train.txt", sep=" ")
#df = pd.read_csv("data/dev.txt", sep=" ")
#df = pd.read_csv("data/test.txt", sep=" ")

df.drop(['-X-', '-X-.1'], axis=1, inplace=True)

df.columns = ['Word', 'Tag']

df.drop(df[df['Word'] == "-DOCSTART-"].index, axis=0, inplace=True)

tags = df['Tag'].values
words = df['Word'].values

new_tags = []
for t in tags:
    if t == "B-ORG":
        new_tags.append("ORG")
    elif t == "B-PER":
        new_tags.append("PER")
    elif t == "I-PER":
        new_tags.append("PER")
    elif t == "B-LOC":
        new_tags.append("LOC")
    elif t == "I-ORG":
        new_tags.append("ORG")
    elif t == "I-LOC":
        new_tags.append("LOC")
    else:
        new_tags.append("O")


df.drop(['Tag'], axis=1, inplace=True)
df.drop(['Word'], axis=1, inplace=True)


df['Tag'] = new_tags
df['Word'] = words

sentence_no = 0
sentence = []
for w in words:
    sentence.append(sentence_no)
    if w == ".":
        sentence_no = sentence_no + 1

df.insert(0, 'Sentence #', sentence)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_whole_word_mask=True)

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(df)
sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
labels = [[s[1] for s in sentence] for sentence in getter.sentences]

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(str(word))
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, labels)]
#tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
#labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]


tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]


i = 0
sentence_no = 0
new_sentence = []
new_data = []
for sentence, label in zip(tokenized_texts, labels):
    new_tokens = []
    new_tags = []
    for token, tag in zip(sentence, label):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_tokens.append(token)
            new_tags.append(tag)
    for new_token, new_tag in zip(new_tokens, new_tags):
        new_data.append((sentence_no, new_token, new_tag))
        print(new_data)
    if i == 20:
        break
    i = i + 1
    new_sentence.append(new_tokens)
    sentence_no = sentence_no + 1