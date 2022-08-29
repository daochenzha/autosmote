import os
import argparse
import pickle
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import logging
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


from autosmote.data_loading import get_data
from autosmote.classifiers import get_clf
from autosmote.rl.training import train

def get_parser():
    parser = argparse.ArgumentParser(description='AutoSMOTE')

    parser.add_argument(
        "--dataset",
        default="mozilla4",
        choices=[
            "phoneme",
            "eeg-eye-state",
            "mozilla4",
            "electricity",
            "MagicTelescope",
            "PhishingWebsites",
        ],
        type=str
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int
    )
    parser.add_argument(
        "--clf",
        default="svm",
        type=str,
        choices=["knn", "svm", "dt", "adaboost"]
    )
    parser.add_argument(
        "--metric",
        default="macro_f1",
        type=str,
        choices=["macro_f1", "mcc"]
    )

    # Args for RL training
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--cuda", default="0", type=str)

    parser.add_argument("--xpid", default="AutoSMOTE",
                        help="Experiment id (default: None).")
    parser.add_argument("--undersample_ratio", default=100, type=int)

    parser.add_argument("--num_instance_specific_actions", default=10, type=int)
    parser.add_argument("--num_max_neighbors", default=30, type=int)
    parser.add_argument("--cross_instance_scale", default=4, type=int)

    # Training settings.
    parser.add_argument("--savedir", default="logs",
                        help="Root dir where experiment data will be saved.")
    parser.add_argument("--num_actors", default=40, type=int, metavar="N",
                        help="Number of actors (default: 20).")
    parser.add_argument("--total_steps", default=1000, type=int, metavar="T",
                        help="Total environment steps to train for.")
    parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                        help="Learner batch size.")
    parser.add_argument("--cross_instance_unroll_length", default=2, type=int, metavar="T",
                        help="The unroll length (time dimension).")
    parser.add_argument("--instance_specific_unroll_length", default=300, type=int, metavar="T",
                        help="The unroll length (time dimension).")
    parser.add_argument("--low_level_unroll_length", default=300, type=int, metavar="T",
                        help="The unroll length (time dimension).")
    parser.add_argument("--num_buffers", default=20, type=int,
                        metavar="N", help="Number of shared-memory buffers.")

    # Loss settings.
    parser.add_argument("--entropy_cost", default=0.0006,
                        type=float, help="Entropy cost/multiplier.")
    parser.add_argument("--baseline_cost", default=0.5,
                        type=float, help="Baseline cost/multiplier.")
    parser.add_argument("--discounting", default=1.0,
                        type=float, help="Discounting factor.")

    # Optimizer settings.
    parser.add_argument("--learning_rate", default=0.005,
                        type=float, metavar="LR", help="Learning rate.")
    parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                        help="Global gradient norm clip.")

    args = parser.parse_args()

    return args

def main(flags):
    train_X, train_y, val_X, val_y, test_X, test_y = get_data(
        name=flags.dataset,
        val_ratio=0.2,
        test_raito=0.2,
        undersample_ratio=flags.undersample_ratio
    )

    clf = get_clf(flags.clf)

    # Search space for ratios
    flags.ratio_map = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Start training
    score = train(flags, train_X, train_y, val_X, val_y, test_X, test_y, clf)

    print("Results:")
    print(flags.dataset, score)

if __name__ == "__main__":
    flags = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = flags.cuda

    # Seed
    np.random.seed(flags.seed)

    main(flags)
