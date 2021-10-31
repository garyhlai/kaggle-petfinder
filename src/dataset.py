import cv2

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # TODO: maybe don't do real time augmentation (slow, augment beforehand instead)
        if self.augmentation:
            augmented = self.augmentation(image=image)
            image = augmented["image"]

        return {
            "image": image,
            "dense_features": self.dense_features[i],
            "target": self.targets[i]
        }