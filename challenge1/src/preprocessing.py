pip install numpy pandas matplotlib scikit-learn opencv-python torch torchvision torchaudio

import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/soil_classification-2025/train_labels.csv')

label_mapping = {
    'Alluvial soil': 0,
    'Black Soil': 1,
    'Clay soil': 2,
    'Red soil': 3
}

df['label_enc'] = df['soil_type'].map(label_mapping)

print(df.head())


import pandas as pd
import os
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm


class SoilDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_id']
        label = int(self.dataframe.iloc[idx]['label_enc'])
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_enc'], random_state=42)



