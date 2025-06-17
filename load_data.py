import os
import kagglehub
from modifier import main

if __name__ == "__main__":
    # Download latest version
    path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

    print("Path to dataset files:", path)
    
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                images.append(os.path.join(root, file))
                
    for image_path in images:
        main(image_path)