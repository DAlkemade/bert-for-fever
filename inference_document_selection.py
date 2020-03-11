import argparse
import os
import pickle
import pprint

import numpy as np
import torch
from transformers import *

from bert_model_wrapper import DocumentBertModel
from inference_util import create_dataloader_dev
from input import load_or_create_evidence, retrieve_claim_doc_ids

N = 5
pp = pprint.PrettyPrinter(indent=4)
OUT_TAG = 'using 50 docs from baseline'


def main():
    args = retrieve_cmd_args()
    features_fname = args.features
    cached_features_file_dev = os.path.join(args.workdir, features_fname)  # nog maken
    data_fname = os.path.join(args.workdir, args.data)  # nog maken
    model_fname = args.model

    claim_ids, doc_ids = retrieve_claim_doc_ids(data_fname)

    print("Load cached dev features")
    features_dev = torch.load(cached_features_file_dev)
    print("Loaded features")
    print(f'Len features: {len(features_dev)}')

    if args.halfdata:
        claim_ids, doc_ids, features_dev = get_half_data(args.firsthalf, claim_ids, doc_ids, features_dev,
                                                         args.splitidx)

    torch.cuda.empty_cache()

    print("Create dev dataloader")
    dataloader_dev = create_dataloader_dev(features_dev, claim_ids)
    model = BertForSequenceClassification.from_pretrained(f"/content/drive/My Drive/Cambridge/L101/{model_fname}",
                                                          num_labels=2)
    bert_model = DocumentBertModel(model)
    evidence, evidence_all_scores = load_or_create_evidence(args.previousevidence, claim_ids)
    print(f"Number of claims: {len(evidence.keys())}")
    print("Done")

    bert_model.predict(dataloader_dev, doc_ids, evidence, evidence_all_scores)

    mean_number_evidence_predictions_per_claim = np.mean([len(value) for value in evidence.values()])
    print(mean_number_evidence_predictions_per_claim)
    with open('/content/drive/My Drive/Overig/docs_evidence_full_hnm.pkl', 'wb') as f:
        pickle.dump(evidence, f)
    with open('/content/drive/My Drive/Overig/docs_evidence_all_full_hnm.pkl', 'wb') as f:
        pickle.dump(evidence_all_scores, f)


def get_half_data(first_half: bool, claim_ids, doc_ids, features_dev, number_instances):
    if first_half:
        features_dev = features_dev[:number_instances]
        claim_ids = claim_ids[:number_instances]
        doc_ids = doc_ids[:number_instances]
    else:
        features_dev = features_dev[number_instances:]
        claim_ids = claim_ids[number_instances:]
        doc_ids = doc_ids[number_instances:]
    return claim_ids, doc_ids, features_dev


def retrieve_cmd_args():
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument("--features", required=True, type=str)
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--docs", required=True, type=str)
    parser.add_argument("--previousevidence", required=True, type=bool)
    parser.add_argument("--firsthalf", required=True, type=bool)
    parser.add_argument("--halfdata", required=True, type=bool)
    parser.add_argument("--workdir", default='/content/drive/My Drive/Overig', type=str)
    parser.add_argument("--splitidx", default=500000, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """"Sorry for the bloated file, it was converted from a notebook."""
    main()
