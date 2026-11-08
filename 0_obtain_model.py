import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

# 2) download the ResNet50 model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
state_dict = torch.hub.load_state_dict_from_url(
    weights.url,
    model_dir=".",   # target directory
    file_name="fasterrcnn_resnet50_fpn_v2_coco.pth",
    progress=True,
)

print("Training done 😄!")