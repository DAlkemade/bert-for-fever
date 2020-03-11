import argparse
import json
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange

from input import parse_doc, get_doc_text

EMPTY_TOKEN = 'EMPTY'


def get_golden_docs_sentences(evidence):
    """Get the golden docs and sentence idxs from the training data file."""
    all_evi = [[e[2], e[3]] for eg in evidence for e in eg if e[3] is not None]  # from baseline scorer
    gold_docs_sentences = {}
    for entry in all_evi:
        id = entry[0]
        sentence_idx = entry[1]
        gold_docs_sentences.setdefault(id, []).append(sentence_idx)

    return gold_docs_sentences


def main():
    """Run sentence preprocessing.

    Retrieve sentences from document selected by the previous step in the pipeline, make ready for tokenization
    and save in a csv file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='D:/GitHubD/fever-allennlp/data/fever/fever.db', type=str)
    parser.add_argument('--infile', default='D:/GitHubD/fever-allennlp/data/fever-data/predictions_doc_dev_bert.jsonl',
                        type=str)
    parser.add_argument('--outfile', default='D:/GitHubD/L101/data/dev_sentences_from_bert_doc_selector.tsv', type=str)
    parser.add_argument('--appendgold', default=False, type=bool)

    args = parser.parse_args()
    db = args.db
    in_file_fname = args.infile
    out_file = args.outfile

    conn = sqlite3.connect(db)
    claim_lengths = []

    training_instances = parse_instances(claim_lengths, conn, in_file_fname, args.appendgold)

    data = pd.DataFrame(training_instances, columns=['label', 'claim', 'context', 'claim_id', 'doc_id', 'sentence_idx'])
    data.to_csv(out_file)
    print(f'Mean claim length: {np.mean(claim_lengths)}')
    plt.hist(claim_lengths, bins=30)
    plt.show()


def parse_instances(claim_lengths, conn, in_file_fname, append_gold: bool):
    """Preprare sentences for next pipeline step.

    Loop through the results of the documents selection step in in_file_fname, retrieve the sentences in these
    documents and suitable format for tokenization.
    """
    with open(in_file_fname, "r") as in_file:
        instances = []
        for line in in_file:
            instances.append(json.loads(line))
        print(f"Number of instances: {len(instances)}")

        training_instances = []
        for i in trange(instances):
            instance = instances[i]

            if instance['verifiable'] != 'NOT VERIFIABLE':
                claim = instance['claim']
                claim_lengths.append(len(claim))
                claim_id = instance['id']
                gold_docs_sentences = get_golden_docs_sentences(instance['evidence'])
                if not append_gold:
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
    return training_instances


if __name__ == '__main__':
    main()
