import os
import pprint

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import *

from inference_util import create_dataloader_dev

def unpack_batch(batch):
    input_ids = batch[0]
    attention_mask = batch[1]
    type_ids = batch[2]
    y_true = batch[3]
    claim_id = batch[4]
    doc_id = [doc_ids[idx] for idx in batch[5]]  #retrieve doc ids
    sentence_idx = [sentence_idxs[idx] for idx in batch[5]]  #retrieve sentence_idx
    return input_ids, attention_mask, type_ids, y_true, claim_id, doc_id, sentence_idx

# Sorry for the bloated file, it was converted from a notebook
if __name__ == '__main__':
    WORK_DIR = '/content/drive/My Drive/Overig'
    features_fname = '200110134032features_include_title=False_from_dev_sentences_from_bert_doc_selector'
    cached_features_file_dev = os.path.join(WORK_DIR, features_fname) # nog maken
    data_fname = '/content/drive/My Drive/Overig/dev_sentences_from_bert_doc_selector.tsv' # nog maken
    model_fname = 'results2ndmodel'
    N = 5
    pp = pprint.PrettyPrinter(indent=4)
    TEST_IDS = [137334, 145446]
    OUT_TAG = 'dev_sentences_on_bert_doc_inputs'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name(0))

    data = pd.read_csv(data_fname)

    print("Load cached dev features")
    features_dev = torch.load(cached_features_file_dev)
    print("Loaded features")
    print(f'Len features: {len(features_dev)}')

    claim_ids = list(data.id)
    doc_ids = list(data.doc_id)
    sentence_idxs = list(data.sentence_idx)

    torch.cuda.empty_cache()
    print("Create dev dataloader")
    dataloader_dev = create_dataloader_dev(features_dev, claim_ids)
    del features_dev

    model = BertForSequenceClassification.from_pretrained(f"/content/drive/My Drive/Cambridge/L101/{model_fname}", num_labels=2)
    model.cuda()
    pass # suppress model.cuda output

    evidence = dict((el,[]) for el in dict.fromkeys(claim_ids))
    evidence_all_scores = dict((el,[]) for el in dict.fromkeys(claim_ids))
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
        input_ids, attention_mask, type_ids, y_true, claim_ids_batch, doc_ids_batch, sentence_idxs_batch = unpack_batch(
            batch)
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
            evidence_all_scores[claim_id.item()].append([softmax_logits[1], doc_ids_batch[i], sentence_idxs_batch[i]])

            # save just the sentences labeled as evidence to dict
            if pred_flat[i] == 1:  # only if classified as evidence
                evidence[claim_id.item()].append([softmax_logits[1], doc_ids_batch[i], sentence_idxs_batch[i]])

    import pickle
    with open(f'/content/drive/My Drive/Overig/sentence_evidence_from{features_fname}.pkl', 'wb') as f:
        pickle.dump(evidence, f)
    with open(f'/content/drive/My Drive/Overig/sentence_evidence_all_from{features_fname}.pkl', 'wb') as f:
        pickle.dump(evidence_all_scores, f)