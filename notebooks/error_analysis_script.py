import pandas as pd
pd.set_option('display.max_rows', 10000)
import numpy as np
from tqdm import tqdm, trange
import spacy
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import transformers
from transformers import BertForTokenClassification, AdamW
from seqeval.metrics import f1_score, accuracy_score
import Levenshtein


tag_values = ['O', 'PER', 'LOC', 'ORG']
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_whole_word_mask=True)

model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)

#Conll03 without gpe conll03_without_gpe_uncase_base_model1.pt
model.load_state_dict(torch.load("../../model/bert_base_conll_4.pt", map_location=torch.device('cpu')), strict=False)

df = pd.read_csv("unprocessed_sampled_asr.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)

df = df[:6723]

g_test = df.groupby("Sentence #")
test_df = pd.DataFrame({"Sentence": g_test.apply(lambda sdf: " ".join(sdf.Word)),
                       "Tag": g_test.apply(lambda sdf: ",".join(sdf.Tag))})

test_df.reset_index(inplace=True)

def model_test(data):
    test = []
    #results = open("conll03_base_ljspeech_asr_test_without_gpe_uncased_results_lower.txt", "a+")
    #test_data=original_data['sentence'].values.tolist()
    #test_data=original_sentence
    #test_data=test_df['Sentence'].values.tolist()
    test_data=data

    # ASR TEST DATE LATEST
    sentence_no = 0
    for data in test_data:
        tokenized_sentence = tokenizer.encode(data.lower().strip())
        #tokenized_sentence = nlp(data.lower().strip())
        input_ids = torch.tensor([tokenized_sentence])
        #input_ids = torch.tensor([tokenized_sentence._.trf_word_pieces])

        with torch.no_grad():
             output = model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

        # join bpe split tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        #tokens = _.trf_word_pieces_
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)

        for token, label in zip(new_tokens, new_labels):
            #result = str(sentence_no) + "\t" + label + "\t" + token + "\n"
            #results.write(result)
            test.append((str(sentence_no), label, token))
        sentence_no = sentence_no + 1
    test_df = pd.DataFrame(test, columns=['sentence_no', 'labels', 'token'])
    return test_df

test_df = model_test(test_df['Sentence'].values.tolist())
#test_df = model_test(asr_df['Sentence'].values.tolist())

indexNames = test_df[test_df['token'] == "[CLS]" ].index
test_df.drop(indexNames, inplace=True)

indexNames = test_df[test_df['token'] == "[SEP]" ].index
test_df.drop(indexNames, inplace=True)

test_df.reset_index(drop=True, inplace=True)

test_df['label_asr'] = df['Tag']
test_df['token_asr'] = df['Word'].str.lower()

def statistics(test_df):
    new_acc = accuracy_score(test_df['labels'].values.tolist(), test_df['label_asr'].values.tolist())
    print(new_acc)

    new_f1 = f1_score(test_df['labels'].values.tolist(), test_df['label_asr'].values.tolist())
    print(new_f1)
    print("---STATISTICS ON EACH LABEL---")
    for tag in ["PER", "LOC", "ORG", "O"]:
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
        print("---------------------------------------")


statistics(test_df)


g_asr = test_df.groupby("sentence_no")
asr_df = pd.DataFrame({'Sentence': g_asr.apply(lambda sdf: " ".join(map(str,sdf.token))),
                      'Tag': g_asr.apply(lambda sdf: ",".join(sdf.labels))})

asr_df['asr_sentence_no'] = asr_df.index
asr_df[["asr_sentence_no"]] = asr_df[["asr_sentence_no"]].apply(pd.to_numeric)
asr_df.sort_values('asr_sentence_no', inplace=True)
asr_df.reset_index(drop=True, inplace=True)

original = pd.read_csv("unprocessed_sampled_original.csv")
original.drop(['Unnamed: 0'], axis=1, inplace=True)
original.tail()

original = original[:7851]

g_original = original.groupby("Sentence #")
original_df = pd.DataFrame({'Sentence': g_original.apply(lambda sdf: " ".join(map(str,sdf.Word))),
                      'Tag': g_original.apply(lambda sdf: ",".join(sdf.Tag))})

original_df.reset_index(inplace=True)

combined_df = pd.DataFrame({"original_sentence": original_df['Sentence'].str.lower(),
                           "original_tags": original_df['Tag'],
                           "asr_sentence": asr_df['Sentence'],
                           "asr_tags": asr_df['Tag']})

def analysis(tag):
#tag = "PER"
    asr_per_ind = []
    original_per_ind = []
    analysis = []
    flag = False
    ignored_sentences = 0
    for i in range(0, len(combined_df), 1):
        sample = combined_df.loc[[i]]
        in_original = False
        for original_sentence, asr_sentence, original_tag, asr_tag in zip(sample['original_sentence'].values.tolist(),
                                                                          sample['asr_sentence'].values.tolist(),
                                                                          sample['original_tags'].values.tolist(),
                                                                          sample['asr_tags'].values.tolist()):
            original_tag_token = np.array(original_tag.split(","))
            asr_tag_token = np.array(asr_tag.split(","))
            original_label = np.array(original_sentence.lower().split())
            asr_label = np.array(asr_sentence.lower().split())

            if tag in original_tag_token:
                original_tag_ind = [index for index, element in enumerate(original_tag_token) if
                                    original_tag_token[index] == tag]
                if tag in asr_tag_token:
                    asr_tag_ind = [index for index, element in enumerate(asr_tag_token) if asr_tag_token[index] == tag]
                    org = " ".join(original_label[original_tag_ind])
                    asr = " ".join(asr_label[asr_tag_ind])
                    error = (1 - (Levenshtein.distance(org, asr) / max(len(org), len(asr)))) * 100
                    analysis.append((i, original_label[original_tag_ind], asr_label[asr_tag_ind], error, True))
                else:
                    check = []
                    o_label = original_label[original_tag_ind]
                    for lab in o_label:
                        j = 0
                        for asr_lab in asr_label:
                            local_error = (1 - (Levenshtein.distance(lab, asr_lab) / max(len(lab), len(asr_lab)))) * 100
                            if local_error >= 50.0:
                                check.append(j)
                            j = j + 1
                    if len(check) > 0:
                        org = " ".join(original_label[original_tag_ind])
                        asr = " ".join(asr_label[check])
                        error = (1 - (Levenshtein.distance(org, asr) / max(len(org), len(asr)))) * 100
                        analysis.append((i, original_label[original_tag_ind], asr_label[check], error, False))
                    else:
                        analysis.append((i, original_label[original_tag_ind], "None", 0.0, False))
    return analysis

analysis_df = pd.DataFrame(analysis("PER"), columns=['Sample #', 'Original', 'ASR', 'Lavenstein', 'Flag'])

orig_asr_found_complete = analysis_df[(analysis_df['Flag'] == True) & (analysis_df['Lavenstein'] == 100.0)]
orig_asr_found_complete_per = (len(orig_asr_found_complete) / len(analysis_df)) * 100

orig_asr_found = analysis_df[(analysis_df['Flag'] == True) & (analysis_df['Lavenstein'] < 100.0) & (analysis_df['Lavenstein'] >= 0.0)]
orig_asr_found_per = (len(orig_asr_found) / len(analysis_df)) * 100

orig_asr_similar = analysis_df[(analysis_df['Flag'] == False) & (analysis_df['Lavenstein'] < 100.0) & (analysis_df['Lavenstein'] > 0.0)]
orig_asr_similar_per = (len(orig_asr_similar) / len(analysis_df)) * 100

orig_asr_nofound = analysis_df[(analysis_df['Flag'] == False) & (analysis_df['Lavenstein'] <= 0.0)]
orig_asr_nofound_per = (len(orig_asr_nofound) / len(analysis_df))*100

def pattern_analysis(sample_df, combined_df):
    ind = np.array(sample_df['Sample #'].values.tolist())
    df = combined_df.loc[ind]
    df.insert(2,'Original',sample_df['Original'].values.tolist())
    df.insert(5,'ASR',sample_df['ASR'].values.tolist())
    df.drop(['original_tags', 'asr_tags'], axis=1, inplace=True)
    df.head(50)
    return df


df_orig_asr_similar = pattern_analysis(orig_asr_similar, combined_df)


def error_sampling(df):
    i = 0
    equal_length_samples = []
    variable_length_samples = []
    for sample, original, asr in zip(df.index,
                                     df['Original'],
                                     df['ASR']):
        if len(original) == len(asr):
            equal_length_samples.append(sample)
        else:
            variable_length_samples.append(sample)
    equal_length_samples.sort()
    variable_length_samples.sort()
    equal_length_samples_df = df.loc[equal_length_samples]
    variable_length_samples_df = df.loc[variable_length_samples]
    return equal_length_samples_df, variable_length_samples_df


equal_length_words_samples_df, variable_length_words_samples_df = error_sampling(df_orig_asr_similar)

def error_sampling2(df):
    check = []
    for sample, original_sentence, asr_sentence, original_tag, asr_tag in zip(
            df.index,
            df['original_sentence'].values.tolist(),
            df['asr_sentence'].values.tolist(),
            df['Original'].values.tolist(),
            df['ASR'].values.tolist()):

        original_label = np.array(original_sentence.split())
        asr_label = np.array(asr_sentence.split())
        original_tag_ind = [index for index, element in enumerate(original_label) if original_label[index] in original_tag]
        asr_tag_ind = [index for index, element in enumerate(asr_label) if asr_label[index] in asr_tag]
        original_bigrams = []
        asr_bigrams = []
        o_label = original_label[original_tag_ind]
        for lab in original_tag:
            for asr_lab in asr_tag:
                local_error = (1 - (Levenshtein.distance(lab, asr_lab) / max(len(lab), len(asr_lab)))) * 100
                if local_error >= 50.0:
                    asr_sentence = asr_sentence.replace(asr_lab, lab)
        check.append((sample, asr_sentence))
    new_asr = pd.DataFrame(check)
    return new_asr


def finding_context(df, n_grams):
    check = []
    for sample, original_sentence, asr_sentence, original_tag, asr_tag in zip(
                df.index,
                df['original_sentence'].values.tolist(),
                df['asr_sentence'].values.tolist(),
                df['Original'].values.tolist(),
                df['ASR'].values.tolist()):

        original_label = np.array(original_sentence.split())
        asr_label = np.array(asr_sentence.split())
        original_tag_ind = [index for index, element in enumerate(original_label) if original_label[index] in original_tag]
        asr_tag_ind = [index for index, element in enumerate(asr_label) if asr_label[index] in asr_tag]
        original_bigrams = []
        asr_bigrams = []
        for l in original_tag_ind:
            if l <= (len(original_label)-1) - n_grams:
                data = ""
                for c in range(-n_grams, n_grams+1, 1):
                    if l+c >= 0:
                        data = data + original_label[l + c] + " "
                    else:
                        continue
                original_bigrams.append(data)
            else:
                data = ""
                for c in range(-n_grams, 1, 1):
                    if l+c < len(original_label):
                        data = data + original_label[l + c] + " "
                    else:
                        continue
                original_bigrams.append(data)
        for l in asr_tag_ind:
            if l <= (len(asr_label) - 1) - n_grams:
                data = ""
                for c in range(-n_grams, n_grams + 1, 1):
                    if l + c >= 0:
                        data = data + asr_label[l + c] + " "
                    else:
                        continue
                asr_bigrams.append(data)
            else:
                data = ""
                for c in range(-n_grams, 1, 1):
                    if l + c < len(asr_label):
                        data = data + asr_label[l + c] + " "
                    else:
                        continue
                asr_bigrams.append(data)
        check.append((sample, (" | ").join(original_bigrams), original_sentence, original_label[original_tag_ind], (" | ").join(asr_bigrams), asr_sentence, asr_label[asr_tag_ind]))
    context = pd.DataFrame(check)
    context.columns = ['Sample #', 'Original N-Grams', "original_sentence", "Original", "ASR N-Grams", "asr_sentence", "ASR"]
    return context


context = finding_context(variable_length_words_samples_df, 5)

for sample, original_ngrams, original_sentence, asr_ngrams, asr_sentence, original_tag, asr_tag in zip(
        context['Sample #'].values.tolist(),
        context['Original N-Grams'].values.tolist(),
        context['original_sentence'].values.tolist(),
        context['ASR N-Grams'].values.tolist(),
        context['asr_sentence'].values.tolist(),
        context['Original'].values.tolist(),
        context['ASR'].values.tolist()):

    original_ngrams = np.array(original_ngrams.split("|"))
    asr_ngrams = np.array(asr_ngrams.split("|"))
    print(original_ngrams)
    print(asr_ngrams)
    local_errors = []
    errors = []
    for _original in original_tag:
        for _asr in asr_ngrams:
            local_error = (1 - (Levenshtein.distance(_original, _asr) / max(len(_original), len(_asr)))) * 100
            local_errors.append(local_error)
        errors.append(local_errors.index(min(local_errors)))
        print(errors)












