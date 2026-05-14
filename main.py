import argparse
from pprint import pprint

import argparse
from pprint import pprint

from src.evaluate import load_and_evaluate
from src.predict import predict_image
from src.train import train
from src.utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Brain Tumor MRI Classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], default="train")
    parser.add_argument(
        "--architecture",
        choices=["enhanced_custom_cnn", "vgg19_cnn", "inceptionv3_cnn"]
    )
    parser.add_argument("--weights", help="Path to .keras model for evaluate/predict")
    parser.add_argument("--image", help="Image path for predict mode")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    if args.architecture:
        config["model"]["architecture"] = args.architecture

    if args.mode == "train":
        metrics = train(config, args.architecture)
        pprint(metrics)

    elif args.mode == "evaluate":
        if not args.weights:
            raise ValueError("--weights is required for evaluate mode")
        metrics = load_and_evaluate(args.weights, config)
        pprint(metrics)

    elif args.mode == "predict":
        if not args.weights or not args.image:
            raise ValueError("--weights and --image are required for predict mode")

        result = predict_image(
            config,
            args.weights,
            args.image,
            config["class_names"],
        )
        pprint(result)


if __name__ == "__main__":
    main()