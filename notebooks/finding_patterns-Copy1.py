import pandas as pd
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)
pd.set_option('max_colwidth', 10000)
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
import string
from punctuator import Punctuator
from fastpunct import FastPunct


tag_values = ['O', 'PER', 'LOC', 'ORG']
#tag_values = ['B-ORG', 'O', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC']
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_whole_word_mask=True)
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx), output_attentions = False, output_hidden_states = False)
model.load_state_dict(torch.load("../../model/bert_base_conll_50.pt", map_location=torch.device('cpu')), strict=False)
p = Punctuator('../../model/INTERSPEECH-T-BRNN.pcl')
# The default language is 'en'
fastpunct = FastPunct('en')


def prepare_data_for_test(filepath):
    df = pd.read_csv(filepath)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df = df[:6723]
    g_test = df.groupby("Sentence #")
    test_df = pd.DataFrame({"Sentence": g_test.apply(lambda sdf: " ".join(sdf.Word)),
                       "Tag": g_test.apply(lambda sdf: ",".join(sdf.Tag))})
    test_df.reset_index(inplace=True)
    return df, test_df


def model_test(data, tokenizer, model):
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
    #test_df.to_csv("final_asr_test_dataframe.csv", index=False)


def prepare_model_output(test_df, df):
    indexNames = test_df[test_df['token'] == "[CLS]" ].index
    test_df.drop(indexNames, inplace=True)
    indexNames = test_df[test_df['token'] == "[SEP]" ].index
    test_df.drop(indexNames, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    test_df['label_asr'] = df['Tag']
    test_df['token_asr'] = df['Word'].str.lower()
    return test_df


def statistics(test_df, tags):
    new_acc = accuracy_score(test_df['labels'].values.tolist(), test_df['label_asr'].values.tolist())
    print(new_acc)

    new_f1 = f1_score(test_df['labels'].values.tolist(), test_df['label_asr'].values.tolist())
    print(new_f1)
    print("---STATISTICS ON EACH LABEL---")
    for tag in tags:
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


df, test_df = prepare_data_for_test('unprocessed_sampled_asr.csv')


#test_df = model_test(new_test, tokenizer, model)
test_df = model_test(test_df['Sentence'].values.tolist(), tokenizer, model)


#test_df = prepare_model_output(test_df, new_df)
test_df = prepare_model_output(test_df, df)


statistics(test_df, ['PER', 'ORG', 'LOC', 'O'])
#0.7758389261744967 without punctuation
#0.676056338028169 with punctuation 1


def prepare_data_for_analysis(test_df, original_data_path):
    g_asr = test_df.groupby("sentence_no")
    asr_df = pd.DataFrame({'Sentence': g_asr.apply(lambda sdf: " ".join(map(str,sdf.token))),
                      'Tag': g_asr.apply(lambda sdf: ",".join(sdf.labels))})
    asr_df['asr_sentence_no'] = asr_df.index
    asr_df[["asr_sentence_no"]] = asr_df[["asr_sentence_no"]].apply(pd.to_numeric)
    asr_df.sort_values('asr_sentence_no', inplace=True)
    asr_df.reset_index(drop=True, inplace=True)
    original = pd.read_csv(original_data_path)
    original.drop(['Unnamed: 0'], axis=1, inplace=True)
    original = original[:7851]
    g_original = original.groupby("Sentence #")
    original_df = pd.DataFrame({'Sentence': g_original.apply(lambda sdf: " ".join(map(str,sdf.Word))),
                      'Tag': g_original.apply(lambda sdf: ",".join(sdf.Tag))})
    original_df.reset_index(inplace=True)
    combined_df = pd.DataFrame({"original_sentence": original_df['Sentence'].str.lower(),
                           "original_tags": original_df['Tag'],
                           "asr_sentence": asr_df['Sentence'],
                           "asr_tags": asr_df['Tag']})
    return asr_df, combined_df

import difflib
def pattern_finding(tag, combined_df):
#tag = "PER"
    analysis = []
    for i in range(0, len(combined_df), 1):
        sample = combined_df.loc[[i]]
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
                    asr_tag_ind = [index for index, element in enumerate(asr_tag_token) if
                                       asr_tag_token[index] == tag]
                    if len(original_tag_ind) < len(asr_tag_ind):
                        org = " ".join(original_label[original_tag_ind])
                        asr = " ".join(asr_label[asr_tag_ind])
                        error = (1 - (Levenshtein.distance(org, asr) / max(len(org), len(asr)))) * 100
                        analysis.append((i, original_label[original_tag_ind], asr_label[asr_tag_ind], error, True))
                    else:
                        asr_tokens = []
                        original_tokens = []
                        errors = []
                        # Sweynheim pannartz
                        # Swain heim pannartz
                        for ind in original_tag_ind:
                            original_entity = original_label[ind]
                            asr_entity = difflib.get_close_matches(original_entity, asr_label[asr_tag_ind])
                            if len(asr_entity) > 0:
                                asr_entity = asr_entity[0]
                                error = (1 - (Levenshtein.distance(original_entity, asr_entity) / max(len(original_entity), len(asr_entity)))) * 100
                                if error >= 50:
                                    asr_tokens.append(asr_entity)
                                    original_tokens.append(original_entity)
                                    errors.append(error)
                                else:
                                    asr_tokens.append("None")
                                    original_tokens.append(original_entity)
                                    errors.append(0.0)
                            else:
                                asr_tokens.append("None")
                                original_tokens.append(original_entity)
                                errors.append(0.0)
                        analysis.append((i, original_tokens, asr_tokens, errors, True))
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
                        if len(o_label) < len(check):
                            org = " ".join(original_label[original_tag_ind])
                            asr = " ".join(asr_label[check])
                            error = (1 - (Levenshtein.distance(org, asr) / max(len(org), len(asr)))) * 100
                            analysis.append((i, original_label[original_tag_ind], asr_label[check], error, False))
                        else:
                            asr_tokens = []
                            original_tokens = []
                            errors = []
                            for ind in original_tag_ind:
                                original_entity = original_label[ind]
                                asr_entity = difflib.get_close_matches(original_entity, asr_label[check])
                                if len(asr_entity) > 0:
                                    asr_entity = asr_entity[0]
                                    error = (1 - (Levenshtein.distance(original_entity, asr_entity) / max(
                                    len(original_entity), len(asr_entity)))) * 100
                                    asr_tokens.append(asr_entity)
                                    original_tokens.append(original_entity)
                                    errors.append(error)
                                else:
                                    asr_tokens.append("None")
                                    original_tokens.append(original_entity)
                                    errors.append(0.0)
                            analysis.append((i, original_tokens, asr_tokens, errors, False))
                    else:
                        analysis.append((i, original_label[original_tag_ind], "None", 0.0, False))
    return analysis

asr_df, combined_df = prepare_data_for_analysis(test_df, 'unprocessed_sampled_original.csv')


analysis_df = pd.DataFrame(pattern_finding("PER", combined_df), columns=['Sample #', 'Original', 'ASR', 'Lavenstein', 'Flag'])


orig_asr_found_complete = analysis_df[(analysis_df['Flag'] == True) & (analysis_df['Lavenstein'] == 100.0)]
orig_asr_found_complete_per = (len(orig_asr_found_complete) / len(analysis_df)) * 100
print(orig_asr_found_complete_per)
orig_asr_found_complete.head()