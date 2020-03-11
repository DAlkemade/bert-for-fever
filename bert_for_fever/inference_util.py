import torch
from torch.utils.data import (DataLoader, TensorDataset)

BATCH_SIZE = 32


def create_dataloader_dev(features, claim_ids):
    # The next lines are taken from the example at https://github.com/huggingface/transformers/blob/0cb163865a4c761c226b151283309eedb2b1ca4d/transformers/data/processors/glue.py#L30
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    all_claim_ids = torch.tensor(claim_ids, dtype=torch.long)
    idx = torch.tensor(range(len(features)), dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_claim_ids, idx)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader
