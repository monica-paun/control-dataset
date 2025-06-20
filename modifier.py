import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101

def get_person_mask(image):
    model = deeplabv3_resnet101(pretrained=True).eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(520),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()
   
    person_mask = (mask == 15).astype(np.uint8)
  
    person_mask = cv2.resize(person_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return person_mask

def apply_transform_on_person(image, transform_fn):
    mask = get_person_mask(image)
    h, w = image.shape[:2]
    pad = max(h, w) // 2  

    image_padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255,255,255])
    mask_padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    person = np.where(np.stack([mask_padded]*3, axis=-1) == 1, image_padded, 255)

    transformed_person = transform_fn(person)
    
    mask_padded_channels = np.stack([mask_padded]*3, axis=-1) * 255
    transformed_mask_channels = transform_fn(mask_padded_channels)
    transformed_mask = (transformed_mask_channels[:,:,0] > 127).astype(np.uint8)  

    start_y, end_y = pad, pad + h
    start_x, end_x = pad, pad + w
    transformed_cropped = transformed_person[start_y:end_y, start_x:end_x]
    mask_cropped = transformed_mask[start_y:end_y, start_x:end_x]

    mask_channels = np.stack([mask_cropped]*3, axis=-1)
    white_bg = np.ones_like(image, dtype=np.uint8) * 255
    result = np.where(mask_channels == 1, transformed_cropped, white_bg)
    return result

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def zoom_image(image, zoom_factor):
    if zoom_factor == 1:
        return image
    h, w = image.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if zoom_factor > 1:
        startx = new_w//2 - w//2
        starty = new_h//2 - h//2
        cropped = resized[starty:starty+h, startx:startx+w]
        return cropped
    else:
        pad_x = (w - new_w) // 2
        pad_y = (h - new_h) // 2
        padded = cv2.copyMakeBorder(resized, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT)
        return padded

def mirror_image(image):
    return cv2.flip(image, 1)

def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")

    image_name = image_path.split("/")[-1]
    cv2.imwrite(f"{image_name}_original.jpg", image)
    transformed_image = apply_transform_on_person(image, lambda x: rotate_image(x, 30))
    cv2.imwrite(f"{image_name}_rotate.jpg", transformed_image)
    transformed_image = apply_transform_on_person(image, lambda x: zoom_image(x, 1.5))
    cv2.imwrite(f"{image_name}_zoom_in.jpg", transformed_image)
    transformed_image = apply_transform_on_person(image, lambda x: zoom_image(x, 0.5))
    cv2.imwrite(f"{image_name}_zoom_out.jpg", transformed_image)
    transformed_image = apply_transform_on_person(image, mirror_image)
    cv2.imwrite(f"{image_name}_mirror.jpg", transformed_image)
    
