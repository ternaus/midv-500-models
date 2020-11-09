from collections import namedtuple

from iglovikov_helper_functions.dl.pytorch.utils import rename_layers
from segmentation_models_pytorch import Unet
from torch import nn
from torch.utils import model_zoo

model = namedtuple("model", ["url", "model"])

models = {
    "Unet_resnet34_2020-05-19": model(
        url="https://github.com/ternaus/midv-500-models/releases/download/0.0.1/unet_resnet34_2020-05-19.zip",
        model=Unet(encoder_name="resnet34", classes=1, encoder_weights=None),
    )
}


def create_model(model_name: str) -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(
        models[model_name].url, progress=True, map_location="cpu"
    )["state_dict"]
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)
    return model
