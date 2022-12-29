import torch

import argparse

from models.branched_network import BranchedNetwork


parser = argparse.ArgumentParser(description="Export parser")
parser.add_argument("--weights_path", type=str,
                    help="Path to the model weights", default="weights/")
parser.add_argument("--model_path", type=str,
                    help="Path to the onnx model", default="models/onnx/model.onnx")


def export(weights_path, model_name):
    img = torch.rand(1, 3, 224, 224)
    measurements = torch.FloatTensor([[25, 0]])
    command = [0]
    dummy_input = (img, measurements, command)

    model = BranchedNetwork(emb_size=128, num_commands=1, num_meas=2)
    model.load_state_dict(torch.load(weights_path))
    torch.onnx.export(model, dummy_input, model_name)


if __name__ == "__main__":
    args = parser.parse_args()
    export(args.weights_path, args.model_path)
