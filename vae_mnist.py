from torchvision import datasets
from torchvision.transforms import v2 as transformsV2

import torch.nn.functional as func
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch


# Define the VAE
class VAEnn(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=400, device='cpu'):
        super(VAEnn, self).__init__()

        self.encoder = nn.Sequential( nn.Flatten(),
                                      nn.Linear(input_dim, hidden_dim),
                                      nn.LeakyReLU(0.2),

                                      nn.Linear(hidden_dim, latent_dim),
                                      nn.LeakyReLU(0.2) )
        
        self.mu = nn.Linear(latent_dim, 2)
        self.red_var = nn.Linear(latent_dim, 2)

        self.decoder = nn.Sequential( nn.Linear(2, latent_dim),
                                      nn.LeakyReLU(0.2),
                                       
                                      nn.Linear(latent_dim, hidden_dim),
                                      nn.LeakyReLU(0.2),
                                      
                                      nn.Linear(hidden_dim, input_dim),
                                      nn.Sigmoid())

    def forward(self, model_input):

        encoder_output = self.encoder(model_input)
        mu, std = self.mu(encoder_output), func.softplus(self.red_var(encoder_output)) + 1e-6
        epsilon = torch.randn_like(mu)
        decoder_input = std*epsilon + mu
        decoder_output = self.decoder(decoder_input)

        return decoder_output, mu, std
    
    def image_regen(self, model_input):

        model_output, _, _ = self.forward(model_input)
        image = torch.unflatten(model_output, 1, (28, 28))

        return image
    
    def latent_regen(self, mu, std):

        epsilon = torch.randn_like(mu)
        decoder_input = std*epsilon + mu
        decoder_output = self.decoder(decoder_input)
        image = self.image_regen(decoder_output)

        return image 

    def trainer(self, train_loader, test_loader, optimizer, epochs):
        
        
        train_loss = -torch.ones(epochs)
        test_loss = -torch.ones(epochs)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            epoch_train_loss = 0
            for batch_idx, (train_batch, label) in enumerate(train_loader):
                decoder_output, mu, std = self.forward(train_batch)
                loss = self.loss_fcn(train_batch, decoder_output, mu, std)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss

            self.eval()
            epoch_test_loss = 0
            for batch_idx, (test_batch, label) in enumerate(test_loader):
                decoder_output, mu, std = self.forward(test_batch)
                loss = self.loss_fcn(test_batch, decoder_output, mu, std)
                epoch_test_loss += loss
                
            train_loss[epoch] = epoch_train_loss
            test_loss[epoch] = epoch_test_loss
        
        return train_loss, test_loss


    def loss_fcn(self, model_input, decoder_output, mu, std):
        model_input_flat = torch.flatten(model_input, start_dim=1)
        reproduction_loss = func.mse_loss(model_input_flat, decoder_output, reduction='mean')
        divergence_loss =  ( (std.pow(2)+mu.pow(2))/2 - torch.log(std) - 1/2 ).sum(1).mean()

        combined_loss = reproduction_loss + 0.001*divergence_loss

        return combined_loss




# Hyperparameters
device = 'cpu'
batch_size = 100
input_dim = 784
hidden_dim = 400
latent_dim = 400
epochs = 50

# Download MNIST and transofrm to tensors
mnist_path = "./raw_mnist"
transform = transformsV2.Compose([transformsV2.ToTensor()])
mnist_train = datasets.MNIST(mnist_path, transform=transform, train=True, download=True)
mnist_test = datasets.MNIST(mnist_path, transform=transform, train=False, download=True)

train_loader = DataLoader(mnist_train, batch_size=batch_size)
test_loader = DataLoader(mnist_test, batch_size=batch_size)

vae_model = VAEnn(input_dim, hidden_dim, latent_dim, device)
optimizer = Adam(vae_model.parameters(), lr=1e-5)

train_loss, test_loss = vae_model.trainer(train_loader, test_loader, optimizer, epochs)

print(test_loss)

