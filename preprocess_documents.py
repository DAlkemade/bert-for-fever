import argparse
import json
import os
import random
import sqlite3

import pandas as pd
from tqdm import tqdm

from input import get_golden_docs, get_doc_text, parse_doc

OUT_FILE_NAME = 'document_selection_test_n=50'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', type=bool, required=True)
    parser.add_argument('--appendgold', type=bool, required=True)
    parser.add_argument('--testset', type=bool, required=True)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--samplenegative', type=bool, default=False)
    args = parser.parse_args()

    if args.local:
        fever_db = 'fever/fever.db'
        root = 'D:/GitHubD/fever-allennlp/data'
    else:
        from google.colab import drive

        drive.mount('/content/drive')
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
        if args.test:
            instances = instances[:100]
        for i in tqdm(range(len(instances))):
            instance = instances[i]
            if args.testset or instance['verifiable'] != 'NOT VERIFIABLE':
                claim = instance['claim']
                claim_id = instance['id']
                docs = instance['predicted_pages']
                if args.appendgold:
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

                    if not args.testset:
                        if doc_id in gold_docs:
                            label = 1
                        else:
                            label = 0
                    else:
                        label = None
                    training_instances.append([label, claim, context, claim_id, doc_id])

    if args.samplenegative:
        new_instances = []
        for f in training_instances:
            if f[0] == 1:
                new_instances.append(f)
            else:
                # throw away 90% of neg instances at random
                if random.uniform(0, 1) < 0.1:
                    new_instances.append(f)
        training_instances = new_instances

    len(training_instances)

    print(len(training_instances))
    data = pd.DataFrame(training_instances, columns=['label', 'claim', 'context', 'claim_id', 'doc_id'])

    data.to_csv(out_file)
