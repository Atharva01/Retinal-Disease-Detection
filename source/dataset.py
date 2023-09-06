import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomSegmentationDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        super(CustomSegmentationDataset,self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = os.listdir(os.path.join(data_dir,"images"))
        self.mask_list = os.listdir(os.path.join(data_dir,"mask"))
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir,"images",self.image_list[idx])
        mask_path = os.path.join(self.data_dir,"mask",self.image_list[idx])
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        if self.transform:
            image = self.transform(image)
            mask  = self.transform(mask)
        return image,mask   
'''
#Image Transformations

data_transforms = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
    transforms.ToTensor(),
    # The mean and standard deviation values used for normalization of pixel
    # These values are commonly used for ImageNet Dataset
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
# Paths for the train and test dataset directories
train_data_dir = 'datasets_drive/training'
test_data_dir = 'datasets_drive/test'

# Instantiate  Dataset
train_dataset = CustomSegmentationDataset(train_data_dir, transform=data_transforms)
test_dataset = CustomSegmentationDataset(test_data_dir, transform=data_transforms)

# Batch Size

batch_size = 32

# Data Loaders

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
'''