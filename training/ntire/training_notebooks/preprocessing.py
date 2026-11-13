from torchvision.models import ResNet50_Weights

transform = ResNet50_Weights.DEFAULT.transforms()
def preprocess(batch):
    batch["Image"] = [transform(img) for img in batch["Image"]]
    return batch
