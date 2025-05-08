import os
import random
import shutil

# Direktori awal
image_dir = "dataset/images"
label_dir = "dataset/labels"

# Direktori output
output_dir = "dataset"
train_ratio = 0.8  # 80% train, 20% val

images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(images)

split_idx = int(len(images) * train_ratio)
train_images = images[:split_idx]
val_images = images[split_idx:]

for phase in ['train', 'val']:
    os.makedirs(f"{output_dir}/images/{phase}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{phase}", exist_ok=True)

def move_files(images, phase):
    for img in images:
        label_file = img.replace('.jpg', '.txt').replace('.png', '.txt')
        shutil.copy(os.path.join(image_dir, img), os.path.join(output_dir, 'images', phase, img))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(output_dir, 'labels', phase, label_file))

move_files(train_images, 'train')
move_files(val_images, 'val')
