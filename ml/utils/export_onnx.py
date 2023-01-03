import torch

import argparse

from models.branched_network import BranchedNetwork


torch.set_default_tensor_type("torch.FloatTensor")
torch.set_default_tensor_type("torch.cuda.FloatTensor")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Export parser")
parser.add_argument("--weights_path", type=str,
                    help="Path to the model weights", default="weights/")
parser.add_argument("--model_path", type=str,
                    help="Path to the onnx model", default="models/onnx/model.onnx")


def export(weights_path, model_name):
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE, dtype=torch.float32)

    input_names = ["input.img"]
    output_names = ["output.angle", "output.speed"]

    model = BranchedNetwork(emb_size=128).to(DEVICE)
    # model.eval()
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    torch.onnx.export(model, dummy_input, model_name,
                      input_names=input_names, output_names=output_names,
                      opset_version=11, export_params=True, verbose=True)


if __name__ == "__main__":
    args = parser.parse_args()
    export(args.weights_path, args.model_path)
