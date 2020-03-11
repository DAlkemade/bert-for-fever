import os

from transformers import BertForSequenceClassification

from bert_for_fever.bert_model_wrapper import SentenceBertModel
from bert_for_fever.inference_util import create_dataloader_dev
from bert_for_fever.input import retrieve_claim_doc_ids, load_or_create_evidence


def predict_sentences(cached_features_file_dev, data_fname: str, model_fname: str, model_dir: str):
    print("Load cached dev features")
    features_dev = torch.load(cached_features_file_dev)
    print("Loaded features")
    print(f'Len features: {len(features_dev)}')
    claim_ids, docids, sentence_idxs = retrieve_claim_doc_ids(data_fname, True)
    torch.cuda.empty_cache()
    print("Create dev dataloader")
    dataloader_dev = create_dataloader_dev(features_dev, claim_ids)
    evidence, evidence_all_scores = load_or_create_evidence(False, claim_ids)
    model = BertForSequenceClassification.from_pretrained(os.path.join(model_dir, model_fname),
                                                          num_labels=2)
    bert_model = SentenceBertModel(model=model)
    bert_model.predict(dataloader_dev, evidence, evidence_all_scores, sentence_idxs, docids)
    return evidence, evidence_all_scores
