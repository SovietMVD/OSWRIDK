import argparse
import os

import config
from dataset import collect_samples
from train import train
from evaluate import evaluate


def _weights_path(dataset):
    return os.path.join(config.WEIGHTS_DIR, dataset + ".pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    data_root = os.path.join(config.DATA_ROOT, args.dataset)

    if args.train:
        samples = collect_samples(os.path.join(data_root, config.TRAIN_DIR))
        num_classes = max(l for _, l in samples) + 1
        out_path = args.model or _weights_path(args.dataset)
        train(samples, num_classes, out_path, seed=args.seed)
        print("saved model to %s" % out_path)

    if args.evaluate:
        model_path = args.model or _weights_path(args.dataset)
        results = evaluate(model_path, data_root)
        for key, value in results.items():
            print("%s: %.4f" % (key, value))


if __name__ == "__main__":
    main()
