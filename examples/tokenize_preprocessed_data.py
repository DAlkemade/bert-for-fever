import argparse
import os
from datetime import datetime

import torch

from bert_for_fever.tokenization.tokenization import tokenize_features


def main():
    """Tokenize the output of one of the preprocess modules to prepare for either inference or training."""
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument("--addtitle", default=False, type=bool, help="Add title to sentence")
    parser.add_argument("--data", default='/content/drive/My Drive/Overig/dev_sentences_from_bert_doc_selector.tsv',
                        type=str, help='tsv file with document or sentence data')
    parser.add_argument("--workdir", default='/content/drive/My Drive/Overig', type=str)
    args = parser.parse_args()

    print("Create features")
    data_fname = args.data
    features = tokenize_features(data_fname, args.addtitle)

    print("Save features")
    time = datetime.now().strftime("%y%m%d%H%M%S")
    fname = f'{time}features_document_selection_from_{data_fname.split(".")[0].split("/")[-1]}'
    torch.save(features, os.path.join(args.workdir, fname))


if __name__ == '__main__':
    main()
