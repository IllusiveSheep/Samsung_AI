import os
import torchvision.models as models
import torch


def prepare_models(model_hand_name="resnet18", model_dot_hand_name="resnet18", model_path="./"):
    model_hand = models.__dict__[model_hand_name](pretrained=True)
    model_dot_hand = models.__dict__[model_dot_hand_name](pretrained=True)

    model_hand.cpu()
    model_dot_hand.cpu()

    torch.save(model_hand, os.path.join(model_path, f"hand_model{model_hand_name}.pth"))
    torch.save(model_dot_hand, os.path.join(model_path, f"hand_model{model_dot_hand_name}.pth"))

    return model_hand, model_dot_hand
