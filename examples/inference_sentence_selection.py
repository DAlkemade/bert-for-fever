import argparse
import os
import pickle

from bert_for_fever.inference.sentences_inference import predict_sentences


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

    evidence, evidence_all_scores = predict_sentences(cached_features_file_dev, data_fname, model_fname, args.modeldir)

    with open(os.path.join(args.workdir, f'sentence_evidence_from{features_fname}.pkl'), 'wb') as f:
        pickle.dump(evidence, f)
    with open(os.path.join(args.workdir, f'sentence_evidence_all_from{features_fname}.pkl'), 'wb') as f:
        pickle.dump(evidence_all_scores, f)


if __name__ == '__main__':
    """Sorry for the bloated file, it was converted from a notebook."""
    main()
