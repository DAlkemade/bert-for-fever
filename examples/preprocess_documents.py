import argparse
import os

from bert_for_fever.preprocessing.document_preprocessing import preprocess_documents


def main():
    """Run documents preprocessing.

    Retrieve documents selected by the baseline, make ready for tokenization
    and save in a csv file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', type=bool, required=True)
    parser.add_argument('--testset', type=bool, required=True)
    parser.add_argument('--samplenegative', type=bool, default=False)
    parser.add_argument('--infile', default='test_baseline_pages.sentences.p5.s5.jsonl', type=str)
    parser.add_argument('--outfile', default='document_selection_test_n=50', type=str)

    args = parser.parse_args()
    test = args.testset
    if args.local:
        fever_db = 'fever/fever.db'
        root = 'D:/GitHubD/fever-allennlp/data'
    else:
        from google.colab import drive

        drive.mount('/content/drive')
        fever_db = 'fever.db'
        root = '/content/drive/My Drive/Overig/'
    db = os.path.join(root, fever_db)

    in_file_fname = os.path.join(root, args.infile)
    out_file = os.path.join(root, f'{args.outfile}.tsv')
    data = preprocess_documents(db, in_file_fname, test, args.samplenegative)
    data.to_csv(out_file)


if __name__ == '__main__':
    main()
