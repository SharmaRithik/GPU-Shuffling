import ray
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet152_Weights
from PIL import Image

ray.init(address='auto')

s3_uri = "s3://anonymous@air-example-data-2/imagenette2/train/"
ds = ray.data.read_images(s3_uri, mode="RGB")
print(ds.schema())

weights = ResNet152_Weights.DEFAULT
model = models.resnet152(weights=weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(), 
    weights.transforms()
])

single_batch = ds.take(10)
transformed_batch = [transform(Image.fromarray(image['image'])) for image in single_batch]

with torch.inference_mode():
    prediction_results = model(torch.stack(transformed_batch).to(device))
    predicted_classes = prediction_results.argmax(dim=1).cpu()

labels = [weights.meta["categories"][i] for i in predicted_classes]
print(labels)

del model
