import argparse
import os
import pickle
import pprint

import torch
from transformers import *

from bert_model_wrapper import SentenceBertModel
from inference_util import create_dataloader_dev
from input import load_or_create_evidence, retrieve_claim_doc_ids

N = 5
pp = pprint.PrettyPrinter(indent=4)
OUT_TAG = 'dev_sentences_on_bert_doc_inputs'


def retrieve_cmd_args():
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument("--features", required=True, type=str)
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--workdir", default='/content/drive/My Drive/Overig', type=str)
    parser.add_argument("--modeldir", default="/content/drive/My Drive/Cambridge/L101", type=str)

    return parser.parse_args()


def main():
    args = retrieve_cmd_args()
    features_fname = args.features
    cached_features_file_dev = os.path.join(args.workdir, features_fname)
    data_fname = os.path.join(args.workdir, args.data)
    model_fname = args.model
    print("Load cached dev features")
    features_dev = torch.load(cached_features_file_dev)
    print("Loaded features")
    print(f'Len features: {len(features_dev)}')
    claim_ids, docids, sentence_idxs = retrieve_claim_doc_ids(data_fname, True)
    torch.cuda.empty_cache()
    print("Create dev dataloader")
    dataloader_dev = create_dataloader_dev(features_dev, claim_ids)

    evidence, evidence_all_scores = load_or_create_evidence(False, claim_ids)

    model = BertForSequenceClassification.from_pretrained(os.path.join(args.modeldir, model_fname),
                                                          num_labels=2)
    bert_model = SentenceBertModel(model=model)
    bert_model.predict(dataloader_dev, evidence, evidence_all_scores, sentence_idxs, docids)

    with open(os.path.join(args.workdir, f'sentence_evidence_from{features_fname}.pkl'), 'wb') as f:
        pickle.dump(evidence, f)
    with open(os.path.join(args.workdir, f'sentence_evidence_all_from{features_fname}.pkl'), 'wb') as f:
        pickle.dump(evidence_all_scores, f)


if __name__ == '__main__':
    """Sorry for the bloated file, it was converted from a notebook."""
    main()
