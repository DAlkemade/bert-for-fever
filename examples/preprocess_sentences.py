import argparse

import matplotlib.pyplot as plt
import numpy as np

from bert_for_fever.preprocessing.sentence_preprocessing import preprocess_sentences


def main():
    """Run sentence preprocessing.

    Retrieve sentences from document selected by the previous step in the pipeline, make ready for tokenization
    and save in a csv file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='D:/GitHubD/fever-allennlp/data/fever/fever.db', type=str)
    parser.add_argument('--infile', default='D:/GitHubD/fever-allennlp/data/fever-data/predictions_doc_dev_bert.jsonl',
                        type=str)
    parser.add_argument('--outfile', default='D:/GitHubD/L101/data/dev_sentences_from_bert_doc_selector.tsv', type=str)
    parser.add_argument('--appendgold', default=False, type=bool)

    args = parser.parse_args()
    db = args.db
    in_file_fname = args.infile
    out_file = args.outfile

    claim_lengths, data = preprocess_sentences(args, db, in_file_fname)
    data.to_csv(out_file)
    print(f'Mean claim length: {np.mean(claim_lengths)}')
    plt.hist(claim_lengths, bins=30)
    plt.show()


if __name__ == '__main__':
    main()
