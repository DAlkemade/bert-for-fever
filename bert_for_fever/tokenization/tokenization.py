import pandas as pd
from tqdm import tqdm
from transformers import InputFeatures, BertTokenizer

MAX_SENTENCE_LENGTH = 512  # maybe make smaller and then batch size higher
PADDING_TOKEN_TYPE_ID = 0  # take the advice at https://github.com/huggingface/transformers/blob/0cb163865a4c761c226b151283309eedb2b1ca4d/transformers/data/processors/glue.py#L30


def prep_instance(claim, context, label, doc_id, add_title, tokenizer, pad_token):
    """Tokenize a single instance, after formatting it."""
    if add_title:
        context = f'[ {doc_id} ] {context}'
    encodings = tokenizer.encode_plus(claim, context, add_special_tokens=True,
                                      max_length=MAX_SENTENCE_LENGTH)  # I expect this will cut off the documents
    input_ids, token_type_ids = encodings["input_ids"], encodings["token_type_ids"]
    # We mask padding with 0
    attention_mask = [1] * len(input_ids)
    # Pad on the right
    padding_length = MAX_SENTENCE_LENGTH - len(input_ids)
    # The next 3 lines are taken from the example at https://github.com/huggingface/transformers/blob/0cb163865a4c761c226b151283309eedb2b1ca4d/transformers/data/processors/glue.py#L30
    input_ids = input_ids + ([pad_token] * padding_length)
    # We mask padding with 0
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([PADDING_TOKEN_TYPE_ID] * padding_length)
    return InputFeatures(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids, label=label)


def create_features(data: pd.DataFrame, add_title: bool, tokenizer: BertTokenizer):
    """Tokenize the instances in data with the tokenizer."""
    claims = list(data.claim)
    contexts = list(data.context)
    labels = list(data.label)
    doc_ids = list(data.doc_id)
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    features = [prep_instance(claims[i], contexts[i], labels[i], doc_ids[i], add_title, tokenizer, pad_token) for i in
                tqdm(range(len(claims)))]
    return features


def tokenize_features(data_fname: str, add_title: bool):
    data = pd.read_csv(data_fname)
    print(f"Number of training instances: {len(data.index)}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    features = create_features(data, add_title, tokenizer)
    return features
