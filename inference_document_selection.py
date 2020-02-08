import argparse

import pandas as pd
from google.colab import drive
import torch
from transformers import *
import numpy as np
import os
import json
import pprint
from scipy.special import softmax
import pickle

from inference_util import create_dataloader_dev


def unpack_batch(batch):
    input_ids = batch[0]
    attention_mask = batch[1]
    type_ids = batch[2]
    y_true = batch[3]
    claim_id = batch[4]
    doc_id = [doc_ids[idx] for idx in batch[5]]  # retrieve doc ids
    return input_ids, attention_mask, type_ids, y_true, claim_id, doc_id


WORK_DIR = '/content/drive/My Drive/Overig'
N = 5
pp = pprint.PrettyPrinter(indent=4)
OUT_TAG = 'using 50 docs from baseline'

# Sorry for the bloated file, it was converted from a notebook
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument("--features", required=True, type=str)
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--docs", required=True, type=str)
    parser.add_argument("--previousevidence", required=True, type=bool)
    parser.add_argument("--firsthalf", required=True, type=bool)
    parser.add_argument("--halfdata", required=True, type=bool)



    args = parser.parse_args()

    features_fname = args.features
    cached_features_file_dev = os.path.join(WORK_DIR, features_fname)  # nog maken
    data_fname = os.path.join(WORK_DIR, args.data)  # nog maken
    model_fname = args.model
    docs_input_file = os.path.join(WORK_DIR, args.docs)

    with open(docs_input_file, "r") as in_file:
        instances = []
        for line in in_file:
            instances.append(json.loads(line))
    print(len(instances))
    for instance in instances:
        instance.pop('predicted_pages', None)  # drop all predicted sentences, since that's what we're doing

    data = pd.read_csv(data_fname)
    data.head(10)

    claim_ids = list(data.claim_id)
    claims_not_reduced = claim_ids
    doc_ids = list(data.doc_id)
    del data

    print("Load cached dev features")
    features_dev = torch.load(cached_features_file_dev)
    print("Loaded features")
    print(f'Len features: {len(features_dev)}')
    number_instances = 500000

    if args.halfdata:
        if args.firsthalf:
            features_dev = features_dev[:number_instances]
            claim_ids = claim_ids[:number_instances]
            doc_ids = doc_ids[:number_instances]
        else:
            features_dev = features_dev[number_instances:]
            claim_ids = claim_ids[number_instances:]
            doc_ids = doc_ids[number_instances:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    features_length = len(features_dev)
    print(f'Len features after: {features_length}')
    print(f'Len doc ids: {len(doc_ids)}')
    print(f'Len claim ids: {len(claim_ids)}')

    torch.cuda.empty_cache()
    print("Create dev dataloader")
    dataloader_dev = create_dataloader_dev(features_dev, claim_ids)
    del features_dev

    model = BertForSequenceClassification.from_pretrained(f"/content/drive/My Drive/Cambridge/L101/{model_fname}",
                                                          num_labels=2)
    model.cuda()

    if args.previousevidence:
        with open('/content/drive/My Drive/Overig/docs_evidence_full_hnm.pkl', 'rb') as f:
            evidence = pickle.load(f)
        with open('/content/drive/My Drive/Overig/docs_evidence_all_full_hnm.pkl', 'rb') as f:
            evidence_all_scores = pickle.load(f)
        print("Loaded evidence")
    else:
        evidence = dict((el, []) for el in dict.fromkeys(claims_not_reduced))
        evidence_all_scores = dict((el, []) for el in dict.fromkeys(claims_not_reduced))
    print(f"Number of claims: {len(evidence.keys())}")
    print("Done")

    model.eval()

    print("Start evaluation")

    # Variables for evaluation
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    conf_matrix = np.zeros((2, 2))

    print(f"Number of batches: {len(dataloader_dev)}")
    for step, batch in enumerate(dataloader_dev):
        if step % 1000 == 0:
            print(f'\nAt step {step}')
        # Move batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack values
        input_ids, attention_mask, type_ids, y_true, claim_ids_batch, doc_ids_batch = unpack_batch(batch)
        # Telling the model not to compute or store gradients, saving memory and speeding up validation. taken from https://mccormickml.com/2019/07/22/BERT-fine-tuning/
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(input_ids, token_type_ids=type_ids, attention_mask=attention_mask, labels=y_true)
            logits = outputs[1]

        # Move logits and labels to CPU. from https://mccormickml.com/2019/07/22/BERT-fine-tuning/, as this should free up RAM
        logits = logits.detach().cpu().numpy()
        y_true = y_true.to('cpu').numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        y_true_flat = y_true.flatten()

        for i, claim_id in enumerate(claim_ids_batch):
            softmax_logits = softmax(logits[i])  # !!!
            # print(softmax_logits)
            # print(f'For claim {claim_id}; doc_id: {doc_ids_batch[i]}; sentence idx: {sentence_idxs_batch[i]}')
            # save all regression scores to dict
            evidence_all_scores[claim_id.item()].append([softmax_logits[1], doc_ids_batch[i]])

            # save just the sentences labeled as evidence to dict
            if pred_flat[i] == 1:  # only if classified as evidence
                evidence[claim_id.item()].append([softmax_logits[1], doc_ids_batch[i]])

    mean_number_evidence_predictions_per_claim = np.mean([len(value) for value in evidence.values()])
    print(mean_number_evidence_predictions_per_claim)

    with open('/content/drive/My Drive/Overig/docs_evidence_full_hnm.pkl', 'wb') as f:
        pickle.dump(evidence, f)
    with open('/content/drive/My Drive/Overig/docs_evidence_all_full_hnm.pkl', 'wb') as f:
        pickle.dump(evidence_all_scores, f)
