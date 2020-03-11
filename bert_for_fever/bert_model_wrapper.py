import numpy as np
import torch
from scipy.special import softmax
from tqdm import trange
from transformers import BertForSequenceClassification, AdamW

EPOCHS = 1


class BertModelWrapper():
    """"Handles BERT model and its requirements."""

    def __init__(self, model=None):
        if model is None:
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        self.model = model
        self.model.cuda()
        # This is a very standard piece of code for the optimizer parameters, available from many sources
        # e.g. https://worksheets.codalab.org/rest/bundles/0x60ed20fc419641d799f53aa0667d5713/contents/blob/basic_trainer.py
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=2e-5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {torch.cuda.get_device_name(0)}")

    def train(self, dataloader_train):
        """"Train model with data in dataloader_train."""
        train_loss_set = []

        for _ in trange(EPOCHS, desc="Epoch"):

            # TRAINING ON TRAIN SET
            self.model.train()

            # Train the data for one epoch
            for step, batch in enumerate(dataloader_train):
                if step % 1000 == 0:
                    print(f'\nAt step {step}')
                # Move batch to GPU
                batch = tuple(t.to(self.device) for t in batch)
                # Unpack values
                input_ids, attention_mask, type_ids, y_true = unpack_batch_train(batch)
                # Clear out the gradients (by default they accumulate) taken from https://mccormickml.com/2019/07/22/BERT-fine-tuning/
                self.optimizer.zero_grad()
                # Make predictions for the batch
                outputs = self.model(input_ids, token_type_ids=type_ids, attention_mask=attention_mask, labels=y_true)
                loss = outputs[0]
                train_loss_set.append(loss.item())
                loss.backward()  # Backpropagate
                self.optimizer.step()

        return train_loss_set

    def save(self, work_dir: str):
        self.model.save_pretrained(work_dir)


class DocumentBertModel(BertModelWrapper):

    def predict(self, dataloader, doc_ids, evidence: dict, evidence_all_scores: dict):
        """Make predictions for all instances in dataloader and save in place in evidence and evidence_all_scores."""
        self.model.eval()
        print("Start evaluation")

        # Variables for evaluation
        print(f"Number of batches: {len(dataloader)}")
        for step, batch in enumerate(dataloader):
            if step % 1000 == 0:
                print(f'\nAt step {step}')
            # Move batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack values
            input_ids, attention_mask, type_ids, y_true, claim_ids_batch, doc_ids_batch = unpack_batch_document_inference(
                batch, doc_ids)
            # Telling the model not to compute or store gradients, saving memory and speeding up validation. taken from https://mccormickml.com/2019/07/22/BERT-fine-tuning/
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(input_ids, token_type_ids=type_ids, attention_mask=attention_mask, labels=y_true)
                logits = outputs[1]

            # Move logits and labels to CPU. from https://mccormickml.com/2019/07/22/BERT-fine-tuning/, as this should free up RAM
            logits = logits.detach().cpu().numpy()
            y_true = y_true.to('cpu').numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            y_true_flat = y_true.flatten()

            for i, claim_id in enumerate(claim_ids_batch):
                softmax_logits = softmax(logits[i])  # !!!
                # save all regression scores to dict
                evidence_all_scores[claim_id.item()].append([softmax_logits[1], doc_ids_batch[i]])

                # save just the sentences labeled as evidence to dict
                if pred_flat[i] == 1:  # only if classified as evidence
                    evidence[claim_id.item()].append([softmax_logits[1], doc_ids_batch[i]])


class SentenceBertModel(BertModelWrapper):

    def predict(self, dataloader, evidence, evidence_all_scores, sentence_idxs, doc_ids):
        """Make predictions for all instances in dataloader and save in place in evidence and evidence_all_scores."""
        self.model.eval()

        print("Start evaluation")

        # Variables for evaluation
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        conf_matrix = np.zeros((2, 2))

        print(f"Number of batches: {len(dataloader)}")
        for step, batch in enumerate(dataloader):
            if step % 1000 == 0:
                print(f'\nAt step {step}')
            # Move batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack values
            input_ids, attention_mask, type_ids, y_true, claim_ids_batch, doc_ids_batch, sentence_idxs_batch = unpack_batch_sentence_inference(
                batch, sentence_idxs, doc_ids)
            # Telling the model not to compute or store gradients, saving memory and speeding up validation. taken from https://mccormickml.com/2019/07/22/BERT-fine-tuning/
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(input_ids, token_type_ids=type_ids, attention_mask=attention_mask, labels=y_true)
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
                evidence_all_scores[claim_id.item()].append(
                    [softmax_logits[1], doc_ids_batch[i], sentence_idxs_batch[i]])

                # save just the sentences labeled as evidence to dict
                if pred_flat[i] == 1:  # only if classified as evidence
                    evidence[claim_id.item()].append([softmax_logits[1], doc_ids_batch[i], sentence_idxs_batch[i]])


def unpack_batch_train(batch):
    input_ids = batch[0]
    attention_mask = batch[1]
    type_ids = batch[2]
    y_true = batch[3]
    return input_ids, attention_mask, type_ids, y_true


def unpack_batch_document_inference(batch, doc_ids):
    input_ids = batch[0]
    attention_mask = batch[1]
    type_ids = batch[2]
    y_true = batch[3]
    claim_id = batch[4]
    doc_id = [doc_ids[idx] for idx in batch[5]]  # retrieve doc ids
    return input_ids, attention_mask, type_ids, y_true, claim_id, doc_id


def unpack_batch_sentence_inference(batch, doc_ids, sentence_idxs):
    input_ids = batch[0]
    attention_mask = batch[1]
    type_ids = batch[2]
    y_true = batch[3]
    claim_id = batch[4]
    doc_id = [doc_ids[idx] for idx in batch[5]]  # retrieve doc ids
    sentence_idx = [sentence_idxs[idx] for idx in batch[5]]  # retrieve sentence_idx
    return input_ids, attention_mask, type_ids, y_true, claim_id, doc_id, sentence_idx
