from torchvision import datasets
from torchvision.transforms import v2 as transformsV2

import torch.nn.functional as func
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch

import torch.distributions as distr

import matplotlib.pyplot as plt

from PIL import Image


# Define the VAE
class VAEnn(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=400, device='cpu'):
        super(VAEnn, self).__init__()

        self.encoder = nn.Sequential( nn.Flatten(),
                                      nn.Linear(input_dim, hidden_dim),
                                      nn.LeakyReLU(0.2),

                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.LeakyReLU(0.2),

                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.LeakyReLU(0.2),

                                      nn.Linear(hidden_dim, latent_dim),
                                      nn.LeakyReLU(0.2) )
        
        self.mu = nn.Linear(latent_dim, 2)
        self.red_var = nn.Linear(latent_dim, 2)

        self.decoder = nn.Sequential( nn.Linear(2, latent_dim),
                                      nn.LeakyReLU(0.2),
                                       
                                      nn.Linear(latent_dim, hidden_dim),
                                      nn.LeakyReLU(0.2),

                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.LeakyReLU(0.2),
                                     
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.LeakyReLU(0.2),
                                      
                                      nn.Linear(hidden_dim, input_dim),
                                      nn.Sigmoid() )
        
        self.classifier = nn.Sequential( nn.Linear(2, 10),
                                         nn.Softmax() )


    def forward(self, model_input):

        encoder_output = self.encoder(model_input)
        mu, std = self.mu(encoder_output), func.softplus(self.red_var(encoder_output)) + 1e-6
        epsilon = torch.randn_like(mu)
        decoder_input = std*epsilon + mu
        decoder_output = self.decoder(decoder_input)
        classif_output = self.classifier(decoder_input)

        return decoder_output, mu, std, classif_output
    
    def image_regen(self, model_input):

        model_output, _, _, _ = self.forward(model_input)
        image = torch.unflatten(model_output, 1, (28, 28))

        return image
    
    def latent_regen(self, mu, std):

        mu, std = torch.tensor([mu], dtype=torch.float32), torch.tensor([std], dtype=torch.float32)
        epsilon = torch.randn_like(mu)
        decoder_input = std*epsilon + mu
        decoder_output = self.decoder(decoder_input)
        image = torch.unflatten(decoder_output, 1, (28, 28))

        return image 

    def trainer(self, train_loader, test_loader, optimizer, epochs):
        
        
        train_loss = -torch.ones(epochs)
        test_loss = -torch.ones(epochs)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            epoch_train_loss = 0
            for batch_idx, (train_batch, label) in enumerate(train_loader):
                decoder_output, mu, std, classsif_output = self.forward(train_batch)
                loss = self.loss_fcn(train_batch, decoder_output, mu, std, classsif_output, label, epoch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss

            self.eval()
            epoch_test_loss = 0
            for batch_idx, (test_batch, label) in enumerate(test_loader):
                decoder_output, mu, std, classsif_output = self.forward(test_batch)
                loss = self.loss_fcn(test_batch, decoder_output, mu, std, classsif_output, label, epoch)
                epoch_test_loss += loss
                
            train_loss[epoch] = epoch_train_loss
            test_loss[epoch] = epoch_test_loss
        
        return train_loss, test_loss


    def loss_fcn(self, model_input, decoder_output, mu, std,  classsif_output, label, epoch):
        model_input_flat = torch.flatten(model_input, start_dim=1)
        reproduction_loss = func.mse_loss(model_input_flat, decoder_output, reduction='mean')

        prior = distr.Normal(mu, std)
        posterior = distr.Normal(torch.zeros_like(mu), torch.ones_like(std))
        divergence_loss = distr.kl_divergence(prior, posterior).sum(dim=1).mean()
        classif_loss = func.cross_entropy(classsif_output, label)

        beta = 0.001 * min(epoch/2, 50)
        alpha = 2 * max(1/(0.5*epoch+1), 0.5)
        combined_loss = reproduction_loss + beta*divergence_loss + alpha*classif_loss

        return combined_loss




# Hyperparameters
device = 'cpu'
batch_size = 100
input_dim = 784
hidden_dim = 100
latent_dim = 100
epochs = 50

# Download MNIST and transofrm to tensors
mnist_path = "./raw_mnist"
transform = transformsV2.Compose([transformsV2.ToTensor()])
mnist_train = datasets.MNIST(mnist_path, transform=transform, train=True, download=True)
mnist_test = datasets.MNIST(mnist_path, transform=transform, train=False, download=True)

train_loader = DataLoader(mnist_train, batch_size=batch_size)
test_loader = DataLoader(mnist_test, batch_size=batch_size)

vae_model = VAEnn(input_dim, hidden_dim, latent_dim, device)
optimizer = Adam(vae_model.parameters(), lr=5e-5)

train_loss, test_loss = vae_model.trainer(train_loader, test_loader, optimizer, epochs)
span = torch.arange(start=-5, end=5, step=0.2)
span_size = len(span)
grid = torch.cartesian_prod(span, span)

canvas = -torch.ones([28*span_size,28*span_size])
for idx, (mu_1, mu_2) in enumerate(grid):
    image = vae_model.latent_regen([mu_1, mu_2], [1e-8, 1e-8])
    x_pos, y_pos = 28*(idx//span_size), (28*idx)%(28*span_size)
    canvas[x_pos:x_pos+28, y_pos:y_pos+28] = image

final_canvas = transformsV2.functional.to_pil_image(canvas)
final_canvas.show()

mu_1_list, mu_2_list, color= [], [], []
digit_colors = {0: "orange", 1: "black", 2: "tab:green", 3: "tab:red", 4: "tab:purple", \
                5: "tab:brown", 6: "tab:pink", 7:"tab:gray", 8: "blue", 9: "tab:cyan"}
for sample, label in mnist_train:
    _, mu, _, _ = vae_model(sample)
    mu_1, mu_2 =  mu[0][0].item(), mu[0][1].item()
    mu_1_list.append(mu_1)
    mu_2_list.append(mu_2)
    color.append(digit_colors[label])


plt.figure(figsize=(6,6))
plt.scatter(mu_1_list, mu_2_list, c=color, s=2, alpha=0.1)
plt.xlabel("mu_1")
plt.ylabel("mu_2")
plt.title("VAE Latent Space")
plt.savefig("bla.pdf")



label = torch.tensor([8])
z = torch.randn((1,2), requires_grad=True)
z_opt = Adam([z], lr=1e-3)
label_onehot = func.one_hot(label, num_classes=10).to(torch.float32)
for _ in range(10000):
    pred = vae_model.classifier(z)
    loss = func.cross_entropy(pred, label_onehot)
    loss.backward()
    z_opt.step()

