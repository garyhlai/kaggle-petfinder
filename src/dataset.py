import cv2
import numpy as np
import torch

class PawpularDataset: 
    def __init__(self, dense_features, targets, image_paths, augmentation): 
        self.dense_features = dense_features
        self.targets = targets
        self.image_paths = image_paths
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        # get image
        image = cv2.imread(self.image_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # need to convert to RGB bc cv2 is BGR by default

        # TODO: maybe don't do real time augmentation (slow, augment beforehand instead)
        if self.augmentation:
            augmented = self.augmentation(image=image)
            image = augmented["image"]
        # image should be shape <batch_size, 256, 256, 3>
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        x = {
            "image": torch.tensor(image, dtype=torch.float),
            "dense_features": torch.tensor(self.dense_features[i],dtype=torch.float)
        }
        y = torch.tensor(self.targets[i], dtype=torch.float).unsqueeze(-1)
        return x, y 