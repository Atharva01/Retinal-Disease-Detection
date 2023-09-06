import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import UNet  # Import your UNet model from model.py
from dataset import CustomSegmentationDataset  # Import your custom dataset from dataset.py
device = 'cuda' if torch.cuda.is_available() else 'cpu'
new_working_directory = "D:/Pytorch_Projects/Retinal-Disease-Detection/"
os.chdir(new_working_directory)
# Define hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# Create an instance of the U-Net model
model = UNet(in_channels=3, out_channels=1).to(device=device)

# Define a loss function (e.g., binary cross-entropy loss)
criterion = nn.BCEWithLogitsLoss()

# Define an optimizer (e.g., Adam)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Instantiate the dataset

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data_dir = 'datasets_drive/training'
#test_data_dir = 'datasets_drive/test'
print(f"Training Dataset found")



train_dataset = CustomSegmentationDataset(train_data_dir,transform=transform)
#test_dataset = CustomSegmentationDataset(test_data_dir)
print(f"Training Dataset created")


# Data Loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(f"Training Dataset loaded")

print(f"Training initiated")

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]"):
        images, masks = batch

        images = images.to(device)
        masks = masks.to(device)        
        
        # Forward pass
        outputs = model(images)
        
        # Calculate the loss
        loss = criterion(outputs, masks)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {total_loss / len(train_dataloader)}")

# Save the trained model
torch.save(model.state_dict(), 'unet_segmentation_model.pth')
