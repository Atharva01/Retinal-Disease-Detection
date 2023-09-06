import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(CustomSegmentationDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform

        # Use glob to get a list of image and mask file paths
        self.image_list = glob.glob(os.path.join(data_dir, "images", "*.tif"))
        self.mask_list = glob.glob(os.path.join(data_dir, "mask", "*.gif"))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        mask_path = self.mask_list[idx]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        if self.transform:
            image = self.transform(image)
            #mask = self.transform(mask)
        return image, mask
