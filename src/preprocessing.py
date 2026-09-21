from PIL import Image
from torchvision import transforms


image_transform = transforms.Compose([
    transforms.Resize(
        (256, 256),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ToTensor()
])


def preprocess_image(image):

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    image = image_transform(image)

    image = image.unsqueeze(0)

    return image
