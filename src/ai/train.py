import click
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np

class ChipDataset(Dataset):
    def __init__(self, data_dir):
        self.images_dir = os.path.join(data_dir, 'images')
        self.masks_dir = os.path.join(data_dir, 'masks')
        self.chip_ids = [f.replace('.npy', '') for f in os.listdir(self.images_dir) if f.endswith('.npy')]
    
    def __len__(self):
        return len(self.chip_ids)
    
    def __getitem__(self, idx):
        chip_id = self.chip_ids[idx]
        
        image = np.load(os.path.join(self.images_dir, f'{chip_id}.npy'))
        mask = np.load(os.path.join(self.masks_dir, f'{chip_id}.npy'))
        
        # Convert to torch tensors and permute to CHW
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask

class SimpleUNet(pl.LightningModule):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, num_classes, 1)
        self.pool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2)
        
    def forward(self, x):
        x1 = torch.relu(self.conv1(x))
        x2 = self.pool(x1)
        x2 = torch.relu(self.conv2(x2))
        x3 = self.upsample(x2)
        x3 = torch.relu(self.conv3(x3))
        return self.conv4(x3)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = torch.nn.functional.cross_entropy(outputs, masks)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

@click.command()
@click.option('--data_dir', required=True, help='Directory with chips')
@click.option('--epochs', default=2, help='Number of epochs')
@click.option('--batch', default=8, help='Batch size')
@click.option('--ckpt_dir', default='outputs/checkpoints', help='Checkpoint directory')
def main(data_dir, epochs, batch, ckpt_dir):
    """Train simple UNet on chips."""
    
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Dataset and dataloader
    dataset = ChipDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    
    # Model
    model = SimpleUNet()
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        default_root_dir=ckpt_dir,
        enable_checkpointing=True
    )
    
    # Train
    trainer.fit(model, dataloader)
    
    # Save best checkpoint
    best_path = os.path.join(ckpt_dir, 'best.ckpt')
    trainer.save_checkpoint(best_path)
    
    print(f"Training complete. Best checkpoint: {best_path}")

if __name__ == '__main__':
    main()