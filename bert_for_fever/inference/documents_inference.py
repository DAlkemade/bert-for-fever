from transformers import BertForSequenceClassification

from bert_for_fever.bert_model_wrapper import DocumentBertModel
from bert_for_fever.inference_util import create_dataloader_dev
from bert_for_fever.input import retrieve_claim_doc_ids, load_or_create_evidence


def predict_documents(cached_features_file_dev: str, data_fname: str, model_fname: str, previous_evidence,
                      half_data: bool, firsthalf=None, splitidx=None):
    claim_ids, doc_ids = retrieve_claim_doc_ids(data_fname)
    print("Load cached dev features")
    features_dev = torch.load(cached_features_file_dev)
    print("Loaded features")
    print(f'Len features: {len(features_dev)}')
    if half_data:
        claim_ids, doc_ids, features_dev = get_half_data(firsthalf, claim_ids, doc_ids, features_dev,
                                                         splitidx)
    torch.cuda.empty_cache()
    print("Create dev dataloader")
    dataloader_dev = create_dataloader_dev(features_dev, claim_ids)
    model = BertForSequenceClassification.from_pretrained(f"/content/drive/My Drive/Cambridge/L101/{model_fname}",
                                                          num_labels=2)
    bert_model = DocumentBertModel(model)
    evidence, evidence_all_scores = load_or_create_evidence(previous_evidence, claim_ids)
    print(f"Number of claims: {len(evidence.keys())}")
    print("Done")
    bert_model.predict(dataloader_dev, doc_ids, evidence, evidence_all_scores)
    return evidence, evidence_all_scores


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
