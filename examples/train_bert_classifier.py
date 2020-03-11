import argparse
import os

from bert_for_fever.training.training import train_model


def main():
    """Train a BERT classifier with on the features supplied with the cmd line argument."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--samplenegative', default=True)
    parser.add_argument("--workdir", default='/content/drive/My Drive/Overig', type=str)

    args = parser.parse_args()
    sample_negative_instances = args.samplenegative
    work_dir = args.workdir

    print("Load cached training features")
    cached_features_file_train = os.path.join(work_dir, args.features)

    train_loss_set = train_model(cached_features_file_train, sample_negative_instances, work_dir)

    with open(os.path.join(work_dir, "losses.txt"), "w") as f:
        for loss in train_loss_set:
            f.write(f'{loss}\n')


# Sorry for the bloated file, it was converted from a notebook
if __name__ == '__main__':
    main()
