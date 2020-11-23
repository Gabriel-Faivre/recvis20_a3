from data import val_data_transforms
import torchvision
import torchvision.models as models
import torch
from torchvision import datasets


od_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
od_model.eval()

val_loader = torch.utils.data.DataLoader(datasets.ImageFolder('bird_dataset/val_images',transform=val_data_transforms),
    batch_size=64, shuffle=False, num_workers=0)

for batch_idx, (data, target) in enumerate(val_loader):
    predictions = od_model(data)
