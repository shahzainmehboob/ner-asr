import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertForTokenClassification, AdamW
from seqeval.metrics import f1_score, accuracy_score

#df = pd.read_csv('conll_preprocessed_train_without_GPE_tags.csv')
#df = pd.read_csv('conll03_original_train.csv')
#dev_df = pd.read_csv("conll_preprocessed_dev_without_GPE_tags.csv")
#dev_df = pd.read_csv('conll03_original_dev.csv')
df = pd.read_csv('data/new_conll_train_preprocessed_with_punctuation_with_broken_entities.csv')
dev_df = pd.read_csv('data/new_conll_dev_preprocessed_with_punctuation_with_broken_entities.csv')
#df.drop(['Unnamed: 0'], axis=1, inplace=True)
#df.drop(df.index[df['Tag'] == 'GPE'], inplace = True)
#df.dropna(inplace=True, axis=0)
#dev_df.drop(['Unnamed: 0'], axis=1, inplace=True)
#dev_df.dropna(inplace=True, axis=0)
#dev_df.drop(dev_df.index[dev_df['Tag'] == 'GPE'], inplace = True)




df['Word'] = df['Word'].str.lower()
dev_df['Word'] = dev_df['Word'].str.lower()


print(len(df))
print(len(dev_df))
print(df['Tag'].unique())
print(df.shape)
print(df.tail())
print(dev_df.tail())
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
dev_getter = SentenceGetter(dev_df)
sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
dev_sentences = [[dev_word[0] for dev_word in dev_sentence] for dev_sentence in dev_getter.sentences]
labels = [[s[1] for s in sentence] for sentence in getter.sentences]
dev_labels = [[dev_s[1] for dev_s in dev_sentence] for dev_sentence in dev_getter.sentences]
tag_values = ['O', 'PER', 'LOC', 'ORG']
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}

print(tag_values)
print(tag2idx)
MAX_LEN = 511
bs = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(n_gpu)
#model_path="/data/users/smehboob/code/examples/ner/ljspeech/language_model"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_whole_word_mask=True)

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
dev_tokenized_texts_and_labels = [tokenize_and_preserve_labels(dev_sent, dev_labs) for dev_sent, dev_labs in zip(dev_sentences, dev_labels)]
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
dev_tokenized_texts = [dev_token_label_pair[0] for dev_token_label_pair in dev_tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
dev_labels = [dev_token_label_pair[1] for dev_token_label_pair in dev_tokenized_texts_and_labels]

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")
dev_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in dev_tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")
dev_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in dev_labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")

attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
dev_attention_masks = [[float(i != 0.0) for i in ii] for ii in dev_input_ids]

#tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, random_state=2018, test_size=0.1)
#tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

tr_inputs, val_inputs, tr_tags, val_tags = input_ids, dev_input_ids, tags, dev_tags
tr_masks, val_masks = attention_masks, dev_attention_masks

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)

model.cuda()

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

from transformers import get_linear_schedule_with_warmup

epochs = 75
max_grad_norm = 1.0

#Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
f1_min_score = 0.0
## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values, accuracy, f1 = [], [], [], []

for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the  completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    acc = accuracy_score(pred_tags, valid_tags)
    f1_scr = f1_score(pred_tags, valid_tags)
    accuracy.append(acc)
    f1.append(f1_scr)
    if f1_scr >= f1_min_score:
        print("Best F1-Score: {}".format(f1_scr))
        torch.save(model.state_dict(), 'bert_base_conll_with_punctuation_with_broken_entities_75.pt')
        f1_min_score = f1_scr
   
    print("Validation Accuracy: {}".format(acc))
    print("Validation F1-Score: {}".format(f1_scr))
    print()

#torch.save(best_model.state_dict(), 'bert_base_conll_50.pt')
print("END")

import matplotlib.pyplot as plt

import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(loss_values, 'b-o', label="training loss")
plt.plot(validation_loss_values, 'r-o', label="validation loss")
plt.plot(accuracy, 'g-o',label="Accuracy")
plt.plot(f1, 'y-o', label="F-1 score")

# Label the plot.
plt.title("Learning curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("bert_base_conll_with_punctuation_with_broken_entities_epoch-loss-conll.png")

'''test_file = open("final_asr_aligned_text.txt", "r")
test_data = test_file.readlines()
#test_data = test_data.replace("\n", " ")
test = [
results = open("conll03_base_asr_test_uncased_results_lower.txt", "a+")


# ASR TEST DATE LATEST
for data in test_data:
    tokenized_sentence = tokenizer.encode(data.lower().strip())
    input_ids = torch.tensor([tokenized_sentence]).to(device)

    with torch.no_grad():
         output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

    # join bpe split tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[-1])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)

    for token, label in zip(new_tokens, new_labels):
        result = label + "\t" + token + "\n"
        results.write(result)
        test.append((label, token))

test_df = pd.DataFrame(test, columns=['labels', 'token'])
#test_df.to_csv("final_asr_test_dataframe.csv", index=False)

#data = data[~data['Word'].str.contains("[CLS]")]
indexNames = test_df[test_df['token'] == "[CLS]" ].index
test_df.drop(indexNames, inplace=True)

#data[data['Word'] == "[CLS]"]
indexNames = test_df[test_df['token'] == "[SEP]" ].index
test_df.drop(indexNames, inplace=True)

test_df.reset_index(drop=True, inplace=True)

data = pd.read_csv("final_asr_processing.csv")
test_df['token_asr'] = data['Word']
test_df['label_asr'] = data['Tag']

new_acc = accuracy_score(test_df['labels'].values.tolist(), test_df['label_asr'].values.tolist())
print(new_acc)

new_f1 = f1_score(test_df['labels'].values.tolist(), test_df['label_asr'].values.tolist())
print(new_f1)
print("---STATISTICS ON EACH LABEL---")
for tag in ["PER", "LOC", "ORG", "GPE", "O"]:
    true_positive = test_df[((test_df['labels'].str.contains(tag)) & (test_df['label_asr'].str.contains(tag)))]
    print(len(true_positive))
    false_positive = test_df[((test_df['labels'].str.contains(tag)) & (~test_df['label_asr'].str.contains(tag)))]
    print(len(false_positive))
    false_negative = test_df[((~test_df['labels'].str.contains(tag)) & (test_df['label_asr'].str.contains(tag)))]
    print(len(false_negative))
    true_negative = test_df[((~test_df['labels'].str.contains(tag)) & (~test_df['label_asr'].str.contains(tag)))]
    print(len(true_negative))
    prec = len(true_positive) / (len(true_positive) + len(false_positive))
    print(prec)
    recall = len(true_positive) / (len(true_positive) + len(false_negative))
    print(recall)
    f_measure = (2 * prec * recall) / (prec + recall)
    print(f_measure)
    print("---------------------------------------")'''


#This is main
'''df_test = pd.read_csv("asr_test_file.txt", header=None)
test_val = df_test[0].values
test = []
for i in test_val:
    test.append(i.split(" "))

df_test_labels = pd.read_csv('asr_test_labels.csv', header=None)
test_labels = df_test_labels[0].values
i = 0
new_tokens, new_labels = [], []

for test_sentence in test:
    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence]).to(device)
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)[0][1:-1]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])[1:-1]
    for token, label_idx in zip(tokens, label_indices):
        new_labels.append(tag_values[label_idx])
        new_tokens.append(token)

    i = i + 1
data = {'test_labels': test_labels, 'pred_labels': new_labels}
df_c = pd.DataFrame(data)

df_c.to_csv("pred_df.csv", index=False)

new_acc = accuracy_score(new_labels, test_labels)
print("Accuracy: " ,new_acc)
new_f1 = f1_score(new_labels, test_labels)
print("F1-Score ", new_f1)'''


















'''data = pd.read_csv("conll_preprocessed_test.csv")
data.drop(["Unnamed: 0"], axis=1, inplace=True)
data.dropna(inplace=True, axis=0)
print(len(data['Tag'].values))
data['Word'] = data['Word'].str.lower()
test_getter = SentenceGetter(data)
sentences = [[word[0] for word in sentence] for sentence in test_getter.sentences]
labels = [[s[1] for s in sentence] for sentence in test_getter.sentences]
tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, labels)]
test = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
test_labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

test = []
results = open("conll03_base_conll03_test_uncased_results_lower.txt", "a+")

i = 0
new_tokens, new_labels = [], []
print("before loop")
for test_sentence in test:
    if len(test_sentence) > 511:
        continue
    tokenized_sentence = tokenizer.encode(test_sentence)
    #tokenized_sentence = tokenized_sentence[:511]
    input_ids = torch.tensor([tokenized_sentence]).to(device)
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)[0][1:-1]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])[1:-1]
    for token, label_idx in zip(tokens, label_indices):
        new_labels.append(tag_values[label_idx])
        new_tokens.append(token)
    for token, label in zip(new_tokens, new_labels):
        result = label + "\t" + token + "\n"
        results.write(result)
        test.append((label, token))

import itertools
test_labels = list(itertools.chain.from_iterable(test_labels))
print(len(new_labels))
print(len(test_labels))

new_acc = accuracy_score(new_labels, test_labels)
print(new_acc)
new_f1 = f1_score(new_labels, test_labels)
print(new_f1)'''

'''test_df = pd.DataFrame(test, columns=['labels', 'token'])
#test_df.to_csv("final_asr_test_dataframe.csv", index=False)

#data = data[~data['Word'].str.contains("[CLS]")]
indexNames = test_df[test_df['token'] == "[CLS]" ].index
test_df.drop(indexNames, inplace=True)

#data[data['Word'] == "[CLS]"]
indexNames = test_df[test_df['token'] == "[SEP]" ].index
test_df.drop(indexNames, inplace=True)

test_df.reset_index(drop=True, inplace=True)

test_df['token_asr'] = data['Word']
test_df['label_asr'] = data['Tag']

new_acc = accuracy_score(test_df['labels'].values.tolist(), test_df['label_asr'].values.tolist())
print(new_acc)

new_f1 = f1_score(test_df['labels'].values.tolist(), test_df['label_asr'].values.tolist())
print(new_f1)
print("---STATISTICS ON EACH LABEL---")
for tag in ["PER", "LOC", "ORG", "GPE", "O"]:
    true_positive = test_df[((test_df['labels'].str.contains(tag)) & (test_df['label_asr'].str.contains(tag)))]
    print(len(true_positive))
    false_positive = test_df[((test_df['labels'].str.contains(tag)) & (~test_df['label_asr'].str.contains(tag)))]
    print(len(false_positive))
    false_negative = test_df[((~test_df['labels'].str.contains(tag)) & (test_df['label_asr'].str.contains(tag)))]
    print(len(false_negative))
    true_negative = test_df[((~test_df['labels'].str.contains(tag)) & (~test_df['label_asr'].str.contains(tag)))]
    print(len(true_negative))
    prec = len(true_positive) / (len(true_positive) + len(false_positive))
    print(prec)
'''

'''i = 0
named_entities = []
new_label_idx = []
for test_sentence in sentences:
    tokenized_sentence = tokenizer.encode(test_sentence)
    if len(tokenized_sentence) > 511:
        tokenized_sentence = tokenized_sentence[:511]
    input_ids = torch.tensor([tokenized_sentence]).to(device)
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        print(token + " --- " +tag_values[label_idx])
        if token.startswith("##"):
            if tag_values[label_idx] == 'O':
                new_tokens[-1] = new_tokens[-1] + token[2:]
                new_labels[-1] = tag_values[label_idx]
            else:
                new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
    print("----------------------------------------------------------------------------")
    for token, label in zip(new_tokens, new_labels):
        new_label_idx.append(tag2idx.get(label))
        named_entities.append(("Sentence: " + str(i), token, label))
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    i = i + 1

pred_df = pd.DataFrame(named_entities)
pred_df.columns = ['Sentence', 'Word', 'Tag']
print(pred_df.head(100))
print(pred_df.tail(100))
print(pred_df[pred_df['Tag'] == "PER"])
print("PRED_DF_SET: "+str(len(pred_df['Tag'].values)))
pred_df.to_csv('conll03_pred_df.csv')

pred_tags = pred_df['Tag'].values
test_tags = test['Tag'].values
test_label_idx = [tag2idx.get(l) for l in test_tags]
new_acc = accuracy_score(new_label_idx, test_label_idx)
print(new_acc)
new_f1 = f1_score(new_label_idx, test_label_idx)
print(new_f1)'''
'''labels = [[s[1] for s in sentence] for sentence in test_getter.sentences]
tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, labels)]
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
tags = [[tag2idx.get(l) for l in lab] for lab in labels]
test_accuracy, test_f1 = [], []
test_labels = []
test_pred = []
named_entities = []
i = 0

for sample in tokenized_texts:
    if len(sample) <= 0:
        continue
    elif len(sample) > 300:
        sample = sample[:300]
    print(sample)
    tokenized_sentence = tokenizer.encode(sample)
    input_ids = torch.tensor([tokenized_sentence]).to(device)
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    test_pred.extend(label_indices)
    test_labels.extend([tags[i]])
    i = i + 1
    # join bpe split tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
    for token, label in zip(new_tokens, new_labels):
        named_entities.append(("Sentence: " + str(i), str(token), str(label)))


pred_df = pd.DataFrame(named_entities)
pred_df.columns = ['Sentence', 'Word', 'Tag']
pred_tags = pred_df['Tag'].values
test_tags = test['Tag'].values
pred_df.to_csv("conll_test_set_pred_df.csv")
new_acc = accuracy_score(test_pred, test_labels)
print(new_acc)
new_f1 = f1_score(test_pred, test_labels)
print(new_f1)
'''

'''test_accuracy, test_f1 = [], []
test_labels = []
named_entities = []
sentence_no = 0                                                                                          
for sample in test_data:                                                                                      
    tokenized_sentence = tokenizer.encode(sample)                                                             
    input_ids = torch.tensor([tokenized_sentence]).to(device)                                                 
    with torch.no_grad():                                                                                     
        output = model(input_ids)                                                                             
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)                                            
    test_labels.append(label_indices)                                                                         
    # join bpe split tokens                                                                                   
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])                                  
    new_tokens, new_labels = [], []                                                                           
    for token, label_idx in zip(tokens, label_indices[0]):                                                    
        if token.startswith("##"):                                                                            
            new_tokens[-1] = new_tokens[-1] + token[2:]                                                       
        else:                                                                                                 
            new_labels.append(tag_values[label_idx])                                                          
            new_tokens.append(token)                                                                          
    for token, label in zip(new_tokens, new_labels):                                                          
        named_entities.append(("Sentence: " + str(sentence_no), token, label))                                
    sentence_no += 1                                                                                          
                                                                                                              
dataframe = pd.DataFrame(named_entities,                                                                      
                            columns=['Sentence #', 'Word', 'Tag'])                                            
dataframe.to_csv("annoated_dataframe-large.csv")'''
