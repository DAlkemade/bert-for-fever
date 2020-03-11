import json
import random
import sqlite3

import pandas as pd
from tqdm import trange

from bert_for_fever.input import get_golden_docs, get_doc_text, parse_doc


def preprocess_documents(db: str, in_file_fname: str, test: bool, sample_negative_instances: bool):
    """Run documents preprocessing.

    Retrieve documents in in_file_name, make ready for tokenization
    """
    conn = sqlite3.connect(db)
    training_instances = parse_instances(test, conn, in_file_fname)
    if sample_negative_instances:
        training_instances = sample_negative(training_instances)
    data = pd.DataFrame(training_instances, columns=['label', 'claim', 'context', 'claim_id', 'doc_id'])
    return data


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


def parse_instances(test_or_dev_set, conn, in_file_fname):
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
            if test_or_dev_set or instance['verifiable'] != 'NOT VERIFIABLE':
                claim = instance['claim']
                claim_id = instance['id']
                docs = instance['predicted_pages']
                if not test_or_dev_set:
                    gold_docs = get_golden_docs(instance['evidence'])
                    for gold_doc in gold_docs:
                        if gold_doc not in docs:
                            docs.append(gold_doc)  # make sure all positive examples are added to the data

                for doc_id in docs:
                    doc_raw = get_doc_text(conn, doc_id)[0]

                    doc_sentences = parse_doc(doc_raw)
                    doc_as_string = ' '.join(doc_sentences)
                    context = doc_as_string

                    if not test_or_dev_set:
                        if doc_id in gold_docs:
                            label = 1
                        else:
                            label = 0
                    else:
                        label = None
                    training_instances.append([label, claim, context, claim_id, doc_id])
    return training_instances
