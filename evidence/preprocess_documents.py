from evidence.input import get_golden_docs, get_doc_text, parse_doc

TEST = False
OUT_FILE_NAME = 'document_selection_test_n=50'
LOCAL = False
SAMPLE_NEGATIVE_INSTANCES = False
APPEND_GOLD_DOCUMENT = False
TEST_SET = True

if not LOCAL:
    from google.colab import drive
    drive.mount('/content/drive')

import argparse
import json
import sqlite3
import pandas as pd
from tqdm import tqdm
import os
import random

if LOCAL:
    fever_db = 'fever/fever.db'
    root = 'D:/GitHubD/fever-allennlp/data'
else:
    fever_db = 'fever.db'
    root = '/content/drive/My Drive/Overig/'

db = os.path.join(root, fever_db)
# in_file_fname = 'D:/GitHubD/fever-allennlp/data/dev_complete.sentences.p5.s5.jsonl'
in_file_fname = os.path.join(root, 'test_baseline_pages.sentences.p5.s5.jsonl')
out_file = os.path.join(root, f'{OUT_FILE_NAME}.tsv')

conn = sqlite3.connect(db)

# TODO: WAT DOEN WE MET DE NOT VERIFIABLES?
with open(in_file_fname, "r") as in_file:
    instances = []
    for line in in_file:
        instances.append(json.loads(line))
    # print(f"Number of instances: {len(instances)}")
    # instances = instances[:75000]

    training_instances = []
    # if TEST:
    #     new_instances = []
    #     for ins in instances:
    #         if ins['id'] == 18884:
    #             new_instances.append(ins)
    #     instances = new_instances
    if TEST:
        instances = instances[:100]
    for i in tqdm(range(len(instances))):
        instance = instances[i]
        if TEST_SET or instance['verifiable'] != 'NOT VERIFIABLE':
            claim = instance['claim']
            claim_id = instance['id']
            docs = instance['predicted_pages']
            if APPEND_GOLD_DOCUMENT:
                gold_docs = get_golden_docs(instance['evidence'])
                for gold_doc in gold_docs:
                    if gold_doc not in docs:
                        docs.append(gold_doc)  # make sure all positive examples are added to the data

            for doc_id in docs:
                doc_raw = get_doc_text(conn, doc_id)[0]

                doc_sentences = parse_doc(doc_raw)
                doc_as_string = ' '.join(doc_sentences)
                doc_as_string_shortened = doc_as_string[:512]
                context = doc_as_string

                if not TEST_SET:
                    if doc_id in gold_docs:
                        label = 1
                    else:
                        label = 0
                else:
                    label = None
                training_instances.append([label, claim, context, claim_id, doc_id])




