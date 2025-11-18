import torch.nn as nn
import torch.nn.functional as func
import torch.distributions as distr
import torch.utils as utils 
import torch.optim as optim
import torch

import numpy as np

from tqdm import tqdm

import pickle as pkl
import gc

class VAE_Model(nn.Module):
    def __init__(self, hyper):
        super(VAE_Model, self).__init__()

        # Define the encoder layers
        # Start with convolutional layers
        encoder_layers = [nn.Conv1d(35, hyper['num_conv_kern'], hyper['size_conv_kern']),\
                          nn.LeakyReLU(0.2)]             
        for _ in range( hyper['num_conv_hidden'] ):
            encoder_layers.append(nn.Conv1d(hyper['num_conv_kern'], hyper['num_conv_kern'], hyper['size_conv_kern']))
            encoder_layers.append(nn.LeakyReLU(0.2))
        
        # Flatten the convolutional layers and FNN
        encoder_layers.append(nn.Flatten())
        flat_layer_size = ( 110-(hyper['size_conv_kern']-1)*(hyper['num_conv_hidden']+1) ) * hyper['num_conv_kern']
        encoder_layers.append(nn.Linear( flat_layer_size, hyper['size_enc_fnn']) )
        encoder_layers.append(nn.LeakyReLU(0.2))
        for _ in range( hyper['num_enc_fnn'] ):
            encoder_layers.append(nn.Linear(hyper['size_enc_fnn'], hyper['size_enc_fnn']))
            encoder_layers.append(nn.LeakyReLU(0.2))

        # Define the mapping to latent space
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(hyper['size_enc_fnn'], hyper['size_latent'])
        self.std_adj = nn.Linear(hyper['size_enc_fnn'], hyper['size_latent'])

        # Define the decoder layers
        decoder_layers = [nn.Linear(hyper['size_latent'],hyper['size_enc_fnn']),\
                          nn.LeakyReLU(0.2)]
        for _ in range( hyper['num_enc_fnn'] ):
            decoder_layers.append(nn.Linear(hyper['size_enc_fnn'], hyper['size_enc_fnn']))
            decoder_layers.append(nn.LeakyReLU(0.2))
        
        self.decoder = nn.Sequential(*decoder_layers)

        # Define the GRU layers
        self.gru = nn.GRU(hyper['size_enc_fnn'], hyper['size_dec_gru'], num_layers=hyper['num_dec_gru'], batch_first=True)
        self.term_gru = nn.GRU(hyper['size_dec_gru']+35, 35, num_layers=hyper['num_dec_gru'], batch_first=True)


        # Define the property predictor
        predictor_layers = [nn.Linear(hyper['size_latent'], hyper['size_pred']),\
                            nn.LeakyReLU(0.2)]
        for _ in range(hyper['num_pred']):
            predictor_layers.append(nn.Linear(hyper['size_pred'], hyper['size_pred']))
            predictor_layers.append(nn.LeakyReLU(0.2))
        predictor_layers.append(nn.Linear(hyper['size_pred'], 3))

        self.predictor = nn.Sequential(*predictor_layers)


    def forward(self, model_input, teacher=False):
        encoder_output = self.encoder(model_input)
        mu, std = self.mu(encoder_output), func.softplus( self.std_adj(encoder_output) )+1e-6
        epsilon = torch.randn_like(mu)
        decoder_input = std*epsilon + mu
        decoder_output = self.decoder(decoder_input)

        decoder_output = decoder_output.unsqueeze(1).repeat([1, 110, 1])
        gru_output, _ = self.gru(decoder_output)

        if teacher:
            termgru_autoreg_in = torch.zeros_like(model_input)
            termgru_autoreg_in[:,:,1:] = model_input[:,:,1:110]
            term_gru_input = torch.cat((gru_output, termgru_autoreg_in.transpose(1,2) ), dim=2)
            term_gru_output, _ = self.term_gru(term_gru_input)
            term_gru_output = term_gru_output.transpose(1,2)
        else:
            term_gru_output = torch.zeros_like(model_input)
            termgru_autoreg_in = torch.zeros([gru_output.shape[0],1,35])
            for idx in range(110):
                term_gru_input = torch.cat((gru_output[:,idx,:].unsqueeze(1), termgru_autoreg_in), dim=2)
                term_gru_output_curr, _ = self.term_gru(term_gru_input)
                termgru_autoreg_in = term_gru_output_curr
                term_gru_output[:,:,idx] = termgru_autoreg_in.squeeze()
        
        predictor_output = self.predictor(mu)
        
        return mu, std, term_gru_output, predictor_output

    def train_n_tester(self, train_loader, test_loader, optimizer, num_epochs):

        train_loss_hist, test_loss_hist = torch.ones(num_epochs), torch.ones(num_epochs)
        for epoch in range(num_epochs):

            # Run the training process for a given epoch
            self.train()
            optimizer.zero_grad()
            for idx, (model_input, predictor_target) in tqdm(enumerate(train_loader), desc=f"Training for epoch {epoch}"):
                mu, std, term_gru_output, predictor_output = self.forward(model_input, teacher=True)
                loss = self.loss_fcn(mu, std, term_gru_output, predictor_output, model_input, predictor_target, epoch)
                loss.backward()
                optimizer.step()
                gc.collect()

            # Check the loss on the train set once the training epoch is done
            self.eval()
            curr_train_loss = 0
            for idx, (model_input, predictor_target) in enumerate(train_loader):
                mu, std, term_gru_output, predictor_output = self.forward(model_input, teacher=False)
                loss = self.loss_fcn(mu, std, term_gru_output, predictor_output, model_input, predictor_target, epoch)
                curr_train_loss += loss.item()
                gc.collect()

            # Check the loss on the test set once the training epoch is done
            curr_test_loss = 0
            for idx, (model_input, predictor_target) in enumerate(test_loader):
                mu, std, term_gru_output, predictor_output = self.forward(model_input, teacher=False)
                loss = self.loss_fcn(mu, std, term_gru_output, predictor_output, model_input, predictor_target, epoch)
                curr_test_loss += loss.item()
                gc.collect()

            train_loss_hist[epoch], test_loss_hist[epoch] = curr_train_loss/len(train_loader), curr_test_loss/len(test_loader)
            print(f"Finished epoch {epoch}, train loss: {train_loss_hist[epoch]:.3g}, test loss: {test_loss_hist[epoch]:.3g}")


        return train_loss_hist, test_loss_hist


    def loss_fcn(self, mu, std, term_gru_output, predictor_output, gru_target, predictor_target, epoch):

        # Reconstruction loss

        rec_loss = func.cross_entropy(term_gru_output, gru_target, reduction='mean')

        # Divergence loss
        prior = distr.Normal(mu, std)
        posterior = distr.Normal(torch.zeros_like(mu), torch.ones_like(std))
        div_loss = distr.kl_divergence(posterior, prior).sum(dim=1).mean()

        # Prediction loss
        pred_loss = func.mse_loss(predictor_output,predictor_target)

        print(rec_loss, div_loss, pred_loss)

        # Annealing coeffs
        alpha = 1
        beta = 0.1

        # Total loss
        loss = rec_loss + alpha * pred_loss + beta * div_loss

        return loss
    
if __name__ == "__main__":

    with open("zinc_train_test.pkl", "rb") as file:
        train_set_df, test_set_df = pkl.load(file)

    num_epochs, batch_size, lr = 2, 100, 1e-6

    hyper_ref = {'num_conv_kern': 8, 'size_conv_kern': 8, 'num_conv_hidden': 4,\
                    'size_enc_fnn': 100, 'num_enc_fnn': 1,  'size_latent': 100,\
                    'size_dec_gru': 50, 'num_dec_gru': 4, 'size_pred': 35,\
                    'num_pred': 3}
    hyper_try = {'num_conv_kern': 8, 'size_conv_kern': 8, 'num_conv_hidden': 4,\
                    'size_enc_fnn': 100, 'num_enc_fnn': 1,  'size_latent': 100,\
                    'size_dec_gru': 50, 'num_dec_gru': 4, 'size_pred': 35,\
                    'num_pred': 3}
    

    vae_model = VAE_Model(hyper_try)
    x_train = torch.tensor( np.array(train_set_df['smiles_hot'].to_list()), dtype=torch.float32)
    y_train = torch.tensor( np.array(train_set_df[['logP', 'qed', 'SAS']].values), dtype=torch.float32)
    del train_set_df
    gc.collect()
    train_set = utils.data.TensorDataset(x_train, y_train)

    x_test = torch.tensor( np.array(test_set_df['smiles_hot'].to_list()), dtype=torch.float32)
    y_test = torch.tensor( np.array(test_set_df[['logP', 'qed', 'SAS']].values), dtype=torch.float32)
    del test_set_df
    gc.collect()
    test_set = utils.data.TensorDataset(x_test, y_test)

    train_loader = utils.data.DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = utils.data.DataLoader(test_set, shuffle=True, batch_size=batch_size)
    
    optimizer = optim.Adam(vae_model.parameters(), lr=lr)
    train_loss_hist, test_loss_hist = vae_model.train_n_tester(train_loader, test_loader, optimizer, num_epochs)

    bla = 1


