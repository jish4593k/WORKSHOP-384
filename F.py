import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import tqdm
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms

def load_images_from_folder(folder):
    images = []
    for filename in tqdm.tqdm(os.listdir(folder), desc="Loading images", ncols=80):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        img = img.resize((299, 299))  # Resize images to fit Inception model input
        img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        images.append(img)
    return np.array(images)

def compute_inception_score(folder, download):
    images = load_images_from_folder(folder)
    
    
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()

    # Load Inception model from TensorFlow Hub
    inception_url = 
    inception_model = hub.load(inception_url)

    
    
    embeddings = inception_model(images_tensor_tf)

    # Compute mean and standard deviation of embeddings
    mean_embeddings = np.mean(embeddings, axis=0)
    std_embeddings = np.std(embeddings, axis=0)

    # Compute Inception Score
    inception_score = np.exp(np.mean(np.sum((embeddings - mean_embeddings) ** 2 / (2 * std_embeddings ** 2) + np.log(std_embeddings) + 0.5)))

    print("Inception Score:", inception_score)

def main():
    parser = argparse.ArgumentParser(description="Compute Inception Score for images.")
    parser.add_argument("--folder", help="Path to the folder of images.")
    parser.add_argument("--download", type=int, default=0, help="Download CIFAR-10 data (1 for True, 0 for False).")
    args = parser.parse_args()

    compute_inception_score(args.folder, args.download)

if __name__ == "__main__":
    main()
