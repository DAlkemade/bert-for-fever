import pandas as pd
from google.colab import drive
import torch
from tqdm import tqdm
from transformers import *
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse

MAX_SENTENCE_LENGTH = 512  # maybe make smaller and then batch size higher
PADDING_TOKEN_TYPE_ID = 0  # take the advice at https://github.com/huggingface/transformers/blob/0cb163865a4c761c226b151283309eedb2b1ca4d/transformers/data/processors/glue.py#L30
WORK_DIR = '/content/drive/My Drive/Overig'


def prep_instance(claim, context, label, doc_id, add_title, tokenizer):
    # print(f'Total length: {len(claim) + len(source_sentence)}')
    if add_title:
        context = f'[ {doc_id} ] {context}'
    encodings = tokenizer.encode_plus(claim, context, add_special_tokens=True,
                                      max_length=MAX_SENTENCE_LENGTH)  # I expect this will cut off the documents
    input_ids, token_type_ids = encodings["input_ids"], encodings["token_type_ids"]
    # We mask padding with 0
    attention_mask = [1] * len(input_ids)
    # Pad on the right
    padding_length = MAX_SENTENCE_LENGTH - len(input_ids)
    # The next 3 lines are taken from the example at https://github.com/huggingface/transformers/blob/0cb163865a4c761c226b151283309eedb2b1ca4d/transformers/data/processors/glue.py#L30
    input_ids = input_ids + ([pad_token] * padding_length)
    # We mask padding with 0
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([PADDING_TOKEN_TYPE_ID] * padding_length)
    return InputFeatures(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids, label=label)


def create_features(data, add_title, tokenizer):
    claims = list(data.claim)
    contexts = list(data.context)
    labels = list(data.label)
    doc_ids = list(data.doc_id)
    features = [prep_instance(claims[i], contexts[i], labels[i], doc_ids[i], add_title, tokenizer) for i in tqdm(range(len(claims)))]
    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument("--test", default=False, type=bool, help="Run on small sample")
    parser.add_argument("--addtitle", default=False, type=bool, help="Add title to sentence")
    parser.add_argument("--data", default='/content/drive/My Drive/Overig/dev_sentences_from_bert_doc_selector.tsv',
                        type=str, help='tsv file with document or sentence data')

    args = parser.parse_args()
    print("Create features")
    drive.mount('/content/drive')
    data_fname = args.data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    data = pd.read_csv(data_fname)

    if args.test:
        data = data[:100]

    print(f"Number of training instances: {len(data.index)}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    features = create_features(data, args.addtitle, tokenizer)
    print("Save features")
    time = datetime.now().strftime("%y%m%d%H%M%S")
    fname = f'{time}features_document_selection_from_{data_fname.split(".")[0].split("/")[-1]}'

    torch.save(features, os.path.join(WORK_DIR, fname))
