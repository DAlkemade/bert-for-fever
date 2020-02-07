from evidence.input import parse_doc, get_doc_text

TEST = False
EMPTY_TOKEN = 'EMPTY'
OUT_FILE_NAME = 'dev_sentences_from_bert_doc_selector'
GOLD = False

import argparse
import json
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

db = 'D:/GitHubD/fever-allennlp/data/fever/fever.db'
# in_file_fname = 'D:/GitHubD/fever-allennlp/data/dev_complete.sentences.p5.s5.jsonl'
in_file_fname = 'D:/GitHubD/fever-allennlp/data/fever-data/predictions_doc_dev_bert.jsonl'
out_file = f'D:/GitHubD/L101/data/{OUT_FILE_NAME}.tsv'

conn = sqlite3.connect(db)

def get_golden_docs_sentences(evidence):
    all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]  # from baseline scorer
    gold_docs_sentences = {}
    for entry in all_evi:
        id = entry[0]
        sentence_idx = entry[1]
        gold_docs_sentences.setdefault(id, []).append(sentence_idx)

    return gold_docs_sentences


# TODO: WAT DOEN WE MET DE NOT VERIFIABLES?
claim_lengths = []
with open(in_file_fname, "r") as in_file:
    instances = []
    for line in in_file:
        instances.append(json.loads(line))
    print(f"Number of instances: {len(instances)}")

    training_instances = []
    if TEST:
        new_instances = []
        for instance in instances:
            if instance['id'] == 137334:
                new_instances.append(instance)
        instances = new_instances
    for step, instance in enumerate(instances):
        if step % 1000 == 0:
            print(f'At step {step}')
        if instance['verifiable'] != 'NOT VERIFIABLE':
            claim = instance['claim']
            claim_lengths.append(len(claim))
            claim_id = instance['id']
            gold_docs_sentences = get_golden_docs_sentences(instance['evidence'])
            if not GOLD:
                docs = instance['predicted_pages']
            else:
                docs = gold_docs_sentences.keys()

            for doc_id in docs:
                doc_sentences = parse_doc(get_doc_text(conn, doc_id)[0])
                if doc_id in gold_docs_sentences.keys():
                    gold_sentences_idx = gold_docs_sentences[doc_id]
                else:
                    gold_sentences_idx = []

                for i in range(len(doc_sentences)):
                    if i in gold_sentences_idx:
                        label = 1
                    else:
                        label = 0
                    sentence = doc_sentences[i]
                    if sentence != EMPTY_TOKEN:
                        training_instances.append([label, claim, sentence, claim_id, doc_id, i])
            # print(instance['evidence'])
            # print(instance['evidence'][0])
            # print(gold_docs_sentences)

    data = pd.DataFrame(training_instances, columns=['evidence', 'claim', 'sentence', 'id', 'doc_id', 'sentence_idx'])

data.to_csv(out_file)

print(np.mean(claim_lengths))
plt.hist(claim_lengths, bins=30)
plt.show()
