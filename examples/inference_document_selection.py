import argparse
import os
import pickle

import numpy as np

from bert_for_fever.inference.documents_inference import predict_documents


def main():
    args = retrieve_cmd_args()
    features_fname = args.features
    cached_features_file_dev = os.path.join(args.workdir, features_fname)  # nog maken
    data_fname = os.path.join(args.workdir, args.data)  # nog maken
    model_fname = args.model

    evidence, evidence_all_scores = predict_documents(cached_features_file_dev, data_fname, model_fname,
                                                      args.previousevidence, firsthalf=args.firsthalf,
                                                      splitidx=args.splitidx)

    mean_number_evidence_predictions_per_claim = np.mean([len(value) for value in evidence.values()])
    print(mean_number_evidence_predictions_per_claim)
    with open('/content/drive/My Drive/Overig/docs_evidence_full_hnm.pkl', 'wb') as f:
        pickle.dump(evidence, f)
    with open('/content/drive/My Drive/Overig/docs_evidence_all_full_hnm.pkl', 'wb') as f:
        pickle.dump(evidence_all_scores, f)


def retrieve_cmd_args():
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument("--features", required=True, type=str)
    parser.add_argument("--data", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--docs", required=True, type=str)
    parser.add_argument("--previousevidence", required=True, type=bool)
    parser.add_argument("--firsthalf", default=None, type=bool)
    parser.add_argument("--halfdata", required=True, type=bool)
    parser.add_argument("--workdir", default='/content/drive/My Drive/Overig', type=str)
    parser.add_argument("--splitidx", default=500000, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """"Sorry for the bloated file, it was converted from a notebook."""
    main()
