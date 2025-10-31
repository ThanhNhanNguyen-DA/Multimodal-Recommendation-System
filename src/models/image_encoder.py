import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from typing import Any, List

class ImageEncoder:
    def __init__(self, pretrained: bool = True):
        self.model = resnet50(pretrained=pretrained)
        self.model.eval()  # Set the model to evaluation mode
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode(self, images: List[Any]) -> torch.Tensor:
        with torch.no_grad():
            transformed_images = [self.transform(image) for image in images]
            batch_tensor = torch.stack(transformed_images)
            embeddings = self.model(batch_tensor)
            return embeddings.numpy()  # Convert to numpy array for compatibility
