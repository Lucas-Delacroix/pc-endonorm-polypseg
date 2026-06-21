import albumentations as A
from albumentations.pytorch import ToTensorV2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SHIFT = (-0.1, 0.1)
SCALE = (0.8, 1.2)
ROTATE = (-10, 10)
PERSPECTIVE_SCALE = (0.05, 0.1)

def affine():
    return A.Affine(translate_percent=SHIFT, scale=SCALE, rotate=ROTATE, p=0.5)

def normalize():
    return A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def get_train_transforms(image_size=352):
    return A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), affine(), A.Perspective(scale=PERSPECTIVE_SCALE, p=0.5), A.GaussNoise(p=0.3), A.Equalize(p=0.3), A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5), normalize(), ToTensorV2()])

def get_val_transforms(image_size=352):
    return A.Compose([A.Resize(image_size, image_size), normalize(), ToTensorV2()])

def get_photometric_transforms():
    return A.Compose([A.GaussNoise(p=0.3), A.Equalize(p=0.3), A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5)])

def get_geometric_transforms(image_size, augment, additional_targets):
    transforms = [A.Resize(image_size, image_size)]
    if augment:
        transforms += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), affine(), A.Perspective(scale=PERSPECTIVE_SCALE, p=0.5)]
    return A.Compose(transforms, additional_targets=additional_targets)
