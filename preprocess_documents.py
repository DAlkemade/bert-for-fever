import argparse
import json
import os
import random
import sqlite3

import pandas as pd
from tqdm import trange

from bert_for_fever.input import get_golden_docs, get_doc_text, parse_doc


def main():
    """Run documents preprocessing.

    Retrieve documents selected by the baseline, make ready for tokenization
    and save in a csv file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', type=bool, required=True)
    parser.add_argument('--testset', type=bool, required=True)
    parser.add_argument('--samplenegative', type=bool, default=False)
    parser.add_argument('--infile', default='test_baseline_pages.sentences.p5.s5.jsonl', type=str)
    parser.add_argument('--outfile', default='document_selection_test_n=50', type=str)

    args = parser.parse_args()
    test = args.testset
    if args.local:
        fever_db = 'fever/fever.db'
        root = 'D:/GitHubD/fever-allennlp/data'
    else:
        from google.colab import drive

        drive.mount('/content/drive')
        fever_db = 'fever.db'
        root = '/content/drive/My Drive/Overig/'
    db = os.path.join(root, fever_db)

    in_file_fname = os.path.join(root, args.infile)
    out_file = os.path.join(root, f'{args.outfile}.tsv')
    conn = sqlite3.connect(db)

    training_instances = parse_instances(test, conn, in_file_fname)
    if args.samplenegative:
        training_instances = sample_negative(training_instances)

    data = pd.DataFrame(training_instances, columns=['label', 'claim', 'context', 'claim_id', 'doc_id'])
    data.to_csv(out_file)


def sample_negative(training_instances):
    new_instances = []
    for f in training_instances:
        if f[0] == 1:
            new_instances.append(f)
        else:
            # throw away 90% of neg instances at random
            if random.uniform(0, 1) < 0.1:
                new_instances.append(f)
    return new_instances


def parse_instances(test, conn, in_file_fname):
    """Preprare documents for next pipeline step.

        Loop through the baselines document selector results in_file_fname, retrieve the text of these documents
        and return in suitable format for tokenization.
        """
    with open(in_file_fname, "r") as in_file:
        instances = []
        for line in in_file:
            instances.append(json.loads(line))

        training_instances = []

        for i in trange(instances):
            instance = instances[i]
            if test or instance['verifiable'] != 'NOT VERIFIABLE':
                claim = instance['claim']
                claim_id = instance['id']
                docs = instance['predicted_pages']
                if not test:
                    gold_docs = get_golden_docs(instance['evidence'])
                    for gold_doc in gold_docs:
                        if gold_doc not in docs:
                            docs.append(gold_doc)  # make sure all positive examples are added to the data

                for doc_id in docs:
                    doc_raw = get_doc_text(conn, doc_id)[0]

                    doc_sentences = parse_doc(doc_raw)
                    doc_as_string = ' '.join(doc_sentences)
                    context = doc_as_string

                    if not test:
                        if doc_id in gold_docs:
                            label = 1
                        else:
                            label = 0
                    else:
                        label = None
                    training_instances.append([label, claim, context, claim_id, doc_id])
    return training_instances


if __name__ == '__main__':
    main()
