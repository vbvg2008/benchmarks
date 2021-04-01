class CachedDataset(Dataset):
    def __init__(self, mscoco_ds):
        self.mscoco_ds = mscoco_ds
        self.images = self.cache_image()

    def __len__(self):
        return len(self.mscoco_ds)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "bbox": self.mscoco_ds[idx]["bbox"]}

    def cache_image(self):
        images = []
        for idx in range(len(self)):
            images.append(cv2.imread(self.mscoco_ds[idx]["image"]))
            if idx % 1000 == 0:
                print("Caching images--{}/{}---".format(idx, len(self.mscoco_ds)))
        return images
