import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from h2q_project.data_loader import CustomDataset  # Assuming data_loader.py exists
from h2q_project.models.simple_gan import Generator, Discriminator  # Assuming simple_gan.py exists
from h2q_project.trainers.base_trainer import BaseTrainer

# Hyperparameters (Move to a config file or command-line arguments later)
LEARNING_RATE = 0.0002
BATCH_SIZE = 64
NUM_EPOCHS = 50

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GANTrainer(BaseTrainer):
    def __init__(self, generator, discriminator, optimizer_g, optimizer_d, criterion, device, train_dataloader, val_dataloader=None):
        super().__init__(generator, optimizer_g, criterion, device, train_dataloader, val_dataloader)
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.real_label = 1
        self.fake_label = 0
        self.epoch = 0 # Track the current epoch


    def train_one_epoch(self, epoch: int):
        self.generator.train()
        self.discriminator.train()
        for i, (inputs, _) in enumerate(self.train_dataloader):
            # Train Discriminator
            self.discriminator.zero_grad()
            real_data = inputs.to(self.device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), self.real_label, dtype=torch.float, device=self.device)
            output = self.discriminator(real_data).view(-1)
            errD_real = self.criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, 100, 1, 1, device=self.device)  # Adjust noise dimensions as needed
            fake = self.generator(noise)
            label.fill_(self.fake_label)
            output = self.discriminator(fake.detach()).view(-1)
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            self.optimizer_d.step()

            # Train Generator
            self.generator.zero_grad()
            label.fill_(self.real_label)
            output = self.discriminator(fake).view(-1)
            errG = self.criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.optimizer_g.step()

            if i % 100 == 0:
                print(f'[{epoch+1}/{NUM_EPOCHS}][{i}/{len(self.train_dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

    def validate(self):
        # GANs are typically not validated in the same way as classifiers.
        # This is a placeholder for potential validation steps, such as visual inspection of generated samples.
        print("GAN validation is not implemented.  Consider visual inspection of generated samples.")
        return 0 #placeholder return

    def train(self, num_epochs: int):
        for epoch in range(self.epoch, num_epochs):
            self.train_one_epoch(epoch)
            # GANs are not typically validated after each epoch, but you could add a validation step here.
            self.epoch += 1 # Increment epoch after each training epoch

    def save_checkpoint(self, path: str):
        torch.save({
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),

        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.epoch = checkpoint.get('epoch', 0) # set epoch to 0 if not exist


def main():
    # Datasets and DataLoaders (Replace with your actual data loading logic)
    train_dataset = CustomDataset(length=1000)  # Example, replace with your data
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    # Loss function
    criterion = nn.BCELoss()

    # Trainer
    trainer = GANTrainer(generator, discriminator, optimizer_g, optimizer_d, criterion, device, train_dataloader)

    # Train the model
    trainer.train(NUM_EPOCHS)

    print('Finished Training')

    # Save the model (Optional)
    trainer.save_checkpoint('gan_model.pth')

if __name__ == '__main__':
    main()
