import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

data_dir = r'C:\Users\admin\OneDrive\Documents\tài liệu\project\machine learning\merged_dog_dataset'
img_size = (32,32)
data =[]
labels = []
miss_image = 0
print("Loading images...")

all_image_paths = []
for label in os.listdir(data_dir):
    data_path = os.path.join(data_dir, label)
    if os.path.isdir(data_path):
        for img_name in os.listdir(data_path):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(data_path,img_name)
                all_image_paths.append((img_path, label))
print(f"Total images found: {len(all_image_paths)}")

for img_path,label in tqdm(all_image_paths, desc="Processing images", unit = 'image'):
    try:
        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: 
            miss_image+=1
            continue
        #resize images
        img = cv2.resize(img, img_size)
        
        #convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #flatten and normalize
        img_vector = img.flatten()
        img_vector = img_vector / 255.0
        
        data.append(img_vector)
        labels.append(label)
    except Exception as e:
        print(f"Error processing image {img_name}: {e}")

df = pd.DataFrame(data)
df['label'] = labels
print("Data loading completed.")
print("Data shape:", df.shape)
print(f"Number of missing images: {miss_image}")

output_file = 'dog_dataset.csv'
df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}") 