import ray
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet152_Weights
from PIL import Image
from typing import Dict
import os

ray.init(address='auto')

s3_uri = "s3://anonymous@air-example-data-2/imagenette2/train/"
ds = ray.data.read_images(s3_uri, mode="RGB")
print(ds.schema())

weights = ResNet152_Weights.DEFAULT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet152(weights=weights).to(device)
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

def preprocess_image(row: Dict[str, np.ndarray]):
    return {
        "original_image": row["image"],
        "transformed_image": transform(Image.fromarray(row["image"])).numpy(),
    }

transformed_ds = ds.map(preprocess_image)

class ResnetModel:
    def __init__(self):
        self.weights = ResNet152_Weights.IMAGENET1K_V1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet152(weights=self.weights).to(self.device)
        self.model.eval()

    def __call__(self, batch: Dict[str, np.ndarray]):
        try:
            torch_batch = torch.stack([torch.tensor(img) for img in batch["transformed_image"]]).to(self.device)
            with torch.inference_mode():
                prediction = self.model(torch_batch)
                predicted_classes = prediction.argmax(dim=1).detach().cpu()
                predicted_labels = [self.weights.meta["categories"][i] for i in predicted_classes]
                return {
                    "predicted_label": predicted_labels,
                    "original_image": batch["original_image"],
                }
        except Exception as e:
            print(f"Error during batch processing: {e}")
            raise

predictions = transformed_ds.map_batches(
    ResnetModel,
    concurrency=1,
    num_gpus=1,
    batch_size=360,
)

prediction_batch = predictions.take_batch(5)

for image, prediction in zip(prediction_batch["original_image"], prediction_batch["predicted_label"]):
    img = Image.fromarray(image)
    img.show()
    print("Label:", prediction)

output_dir = os.path.join(os.getcwd(), "temp")

predictions.drop_columns(["original_image"]).write_parquet(f"local://{output_dir}")
print(f"Predictions saved to `{output_dir}`!")
