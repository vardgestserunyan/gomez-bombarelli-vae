import torch.nn as nn
import torch.nn.functional as func
import torch.distributions as distr
import torch.utils as utils 
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm
import logging
import pickle as pkl
import gc


class VAE_Model(nn.Module):
    def __init__(self, hyper, device):
        super(VAE_Model, self).__init__()

        # Save the hyperparameter settings and the device
        self.hyper = hyper
        self.device = device


        # Define the encoder layers
        # Start with convolutional layers
        encoder_layers = [nn.Conv1d(hyper['alph_size'], hyper['num_conv_kern'], hyper['size_conv_kern']),\
                          nn.LeakyReLU(0.2),\
                          nn.BatchNorm1d(hyper['num_conv_kern'], eps=1e-05)]             
        for _ in range( hyper['num_conv']-1 ):
            encoder_layers.append(nn.Conv1d(hyper['num_conv_kern'], hyper['num_conv_kern'], hyper['size_conv_kern']))
            encoder_layers.append(nn.BatchNorm1d(hyper['num_conv_kern'], eps=1e-05))
            encoder_layers.append(nn.LeakyReLU(0.2))
            
        
        # Flatten the convolutional layers and FNN
        encoder_layers.append(nn.Flatten())
        flat_layer_size = ( 112-(hyper['size_conv_kern']-1)*hyper['num_conv'] ) * hyper['num_conv_kern']
        encoder_layers.append(nn.Linear( flat_layer_size, hyper['size_fnn']) )
        encoder_layers.append(nn.BatchNorm1d(hyper['size_fnn'], eps=1e-05))
        encoder_layers.append(nn.LeakyReLU(0.2))
        for _ in range( hyper['num_fnn']-1 ):
            encoder_layers.append(nn.Linear(hyper['size_fnn'], hyper['size_fnn']))
            encoder_layers.append(nn.BatchNorm1d(hyper['size_fnn'], eps=1e-05))
            encoder_layers.append(nn.LeakyReLU(0.2))

        # Define the mapping to latent space
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(hyper['size_fnn'], hyper['size_latent'])
        self.std_adj = nn.Linear(hyper['size_fnn'], hyper['size_latent'])

        # Define the decoder layers
        decoder_layers = [nn.Linear(hyper['size_latent'],hyper['size_fnn']),\
                          nn.BatchNorm1d(hyper['size_fnn'],  eps=1e-05),\
                          nn.LeakyReLU(0.2)]
        for _ in range( hyper['num_fnn']-2 ):
            decoder_layers.append(nn.Linear(hyper['size_fnn'], hyper['size_fnn']))
            decoder_layers.append(nn.BatchNorm1d(hyper['size_fnn'], eps=1e-05))
            decoder_layers.append(nn.LeakyReLU(0.2))
        decoder_layers.append(nn.Linear(hyper['size_fnn'], hyper['size_gru']))
        self.decoder = nn.Sequential(*decoder_layers)

        # Define the GRU layers
        self.gru = nn.GRU(hyper['alph_size'], hyper['size_gru'], num_layers=hyper['num_gru'], batch_first=True)
        self.post_gru = nn.Linear(hyper['size_gru'], hyper['alph_size'])

        # Define the property predictor
        propreg_layers = [nn.Linear(hyper['size_latent'], hyper['size_propreg']),\
                            nn.BatchNorm1d(hyper['size_propreg'], eps=1e-05),\
                            nn.LeakyReLU(0.2)]
        for _ in range(hyper['num_propreg']-1):
            propreg_layers.append(nn.Linear(hyper['size_propreg'], hyper['size_propreg']))
            propreg_layers.append(nn.BatchNorm1d(hyper['size_propreg'], eps=1e-05))
            propreg_layers.append(nn.LeakyReLU(0.2))
        propreg_layers.append(nn.Linear(hyper['size_propreg'], 3))

        self.propreg = nn.Sequential(*propreg_layers)


    def forward(self, model_input, teacher_prob=0.0):
        encoder_output = self.encoder(model_input)
        mu, std = self.mu(encoder_output), func.softplus( self.std_adj(encoder_output) )+1e-6
        epsilon = torch.randn_like(mu)
        decoder_input = std*epsilon + mu
        decoder_output = (self.decoder(decoder_input)).repeat([self.hyper['num_gru'], 1, 1])

        # Decoder GRUs
        curr_gru_val, curr_hidden = model_input[:,:,0].unsqueeze(1), decoder_output
        gru_output = torch.zeros((model_input.shape[0], model_input.shape[2]-1, model_input.shape[1]), device=model_input.device)
        for idx in range(model_input.shape[2]-1):
            curr_gru_out, curr_hidden = self.gru(curr_gru_val, curr_hidden)
            curr_gru_val = self.post_gru(curr_gru_out)
            gru_output[:,idx,:] = curr_gru_val.squeeze(1)
            
            teacher = ( torch.rand(model_input.shape[0], device=model_input.device) < teacher_prob )
            curr_gru_val_ground = model_input[:,:,idx+1].unsqueeze(1)
            one_hot_idx = curr_gru_val.argmax(dim=2).squeeze(1)
            curr_gru_pred = torch.zeros_like(curr_gru_val, device=model_input.device)
            curr_gru_pred.scatter_(2, one_hot_idx.view(model_input.shape[0], 1, 1), 1)
            curr_gru_val = (~teacher.view(-1,1,1))*curr_gru_pred + teacher.view(-1,1,1)*curr_gru_val_ground

        
        gru_output = gru_output.permute(0,2,1)

        propreg_output = self.propreg(mu)
        
        return mu, std, gru_output, propreg_output

    def train_n_tester(self, train_loader, test_loader, optimizer, num_epochs, lr_scheduler=None):

        logger = logging.getLogger()

        train_loss_hist, test_loss_hist = torch.ones(num_epochs), torch.ones(num_epochs)
        for epoch in range(num_epochs):

            # Run the training process for a given epoch
            self.train()
            logger.info(f"Logging during training process for epoch {epoch}")
            for idx, (model_input, propreg_target) in tqdm(enumerate(train_loader), desc=f"Training for epoch {epoch}"):
                model_input, propreg_target = model_input.to(self.device), propreg_target.to(self.device)
                optimizer.zero_grad()
                teacher_prob = (0.95 + 0.05*func.sigmoid(-0.5*(torch.tensor(epoch)-25)) ).item()
                mu, std, gru_output, propreg_output = self.forward(model_input, teacher_prob=teacher_prob)
                loss = self.loss_fcn(mu, std, gru_output, propreg_output, model_input[:,:,1:], propreg_target, epoch, logger=logger)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1)
                optimizer.step()
                gc.collect()
                if idx == 1 or idx == 2:
                    logger.info("-  -  -  -  -  -")
                    logger.info(f"Predicted: {torch.argmax(gru_output[idx,:,:], dim=0)}")
                    logger.info(f"Ground T: {torch.argmax(model_input[idx,:,:], dim=0)}")
                    logger.info("-  -  -  -  -  -")   
            lr_scheduler.step()           
                
            
            # Check the loss on the train set once the training epoch is done
            with torch.no_grad():
                self.eval()
                curr_train_loss = 0
                logger.info(f"Logging post-train for the train set, epoch {epoch}")
                for idx, (model_input, propreg_target) in enumerate(train_loader):
                    model_input, propreg_target = model_input.to(self.device), propreg_target.to(self.device)
                    mu, std, gru_output, propreg_output = self.forward(model_input, teacher_prob=0.0)
                    loss = self.loss_fcn(mu, std, gru_output, propreg_output, model_input[:,:,1:], propreg_target, epoch, logger=logger)
                    curr_train_loss += loss.item()
                    if idx == 1 or idx == 2:
                        logger.info("-  -  -  -  -  -")
                        logger.info(f"Predicted: {torch.argmax(gru_output[idx,:,:], dim=0)}")
                        logger.info(f"Ground T: {torch.argmax(model_input[idx,:,:], dim=0)}")
                        logger.info("-  -  -  -  -  -")
                    gc.collect()

                
                # Check the loss on the test set once the training epoch is done
                curr_test_loss = 0
                logger.info(f"Logging post-train for the test set, epoch {epoch}")
                for idx, (model_input, propreg_target) in enumerate(test_loader):
                    model_input, propreg_target = model_input.to(self.device), propreg_target.to(self.device)
                    mu, std, gru_output, propreg_output = self.forward(model_input, teacher_prob=0.0)
                    loss = self.loss_fcn(mu, std, gru_output, propreg_output, model_input[:,:,1:], propreg_target, epoch, logger=logger)
                    curr_test_loss += loss.item()
                    gc.collect()

                train_loss_hist[epoch], test_loss_hist[epoch] = curr_train_loss/len(train_loader), curr_test_loss/len(test_loader)
                logger.info(f"Finished epoch {epoch}, train loss: {train_loss_hist[epoch]:.3g}, test loss: {test_loss_hist[epoch]:.3g}")
                logger.info("____________________________________________________________")
            
                

        return train_loss_hist, test_loss_hist


    def loss_fcn(self, mu, std, term_gru_output, predictor_output, gru_target, predictor_target, epoch, logger=False):

        # Reconstruction loss
        gru_target = torch.argmax(gru_target, dim=1)
        rec_loss = func.cross_entropy(term_gru_output, gru_target, reduction='mean')

        # Divergence loss
        prior = distr.Normal(mu, std)
        posterior = distr.Normal(torch.zeros_like(mu), torch.ones_like(std))
        div_loss = distr.kl_divergence(prior, posterior).sum(dim=1).mean()

        # Prediction loss
        pred_loss = func.mse_loss(predictor_output,predictor_target)

        # Annealing coeffs
        alpha = 0.01*func.sigmoid(0.15*(torch.tensor(epoch)-25))
        beta = 0.01*func.sigmoid(0.15*(torch.tensor(epoch)-25))

        if logger:
            logger.info(f"Epoch: {epoch}, Rec: {rec_loss.item():0.4g}, Div: {div_loss.item():.4g}, Pred: {pred_loss.item():0.4g}")

        # Total loss
        loss = (1-alpha-beta)*rec_loss + alpha * pred_loss + beta * div_loss

        return loss
    
    def model_pass_det(self, model_input):

        self.eval()
        with torch.no_grad():
            mu, std, gru_output_raw, propreg_output = self.forward(self, model_input, teacher_prob=0.0)
            gru_output = torch.argmax(gru_output_raw[:,:,:], dim=1)

        return mu, std, gru_output, propreg_output
    
    def model_pass_rand(self, model_input):

        self.eval()
        with torch.no_grad():
            mu, std, gru_output_raw, propreg_output = self.forward(self, model_input, teacher_prob=0.0)
            gru_out_distr = distr.Categorical(probs=gru_output_raw.permute(0,2,1))
            gru_output = gru_out_distr.sample()

        return mu, std, gru_output, propreg_output

    
if __name__ == "__main__":

    logging.basicConfig(filename="train_n_tester.log", level=logging.INFO, format='%(asctime)-15s %(message)s')
    torch.manual_seed(12121995)

    with open("zinc_train_test.pkl", "rb") as file:
        train_set_df, test_set_df, _ = pkl.load(file)

    num_epochs, batch_size, lr = 50, 250, 2e-3
    
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda"

    hyper = {'num_conv_kern': 10, 'size_conv_kern': 10, 'num_conv': 4,\
             'size_fnn': 196, 'num_fnn': 3,  'size_latent': 128,\
             'size_gru': 488, 'num_gru': 4, 'size_propreg': 128,\
             'num_propreg': 2, 'alph_size': 37 }

    vae_model = VAE_Model(hyper, device).to(device)
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
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(0.8**epoch, 0.1))
    train_loss, test_loss = vae_model.train_n_tester(train_loader, test_loader, optimizer, num_epochs, lr_scheduler)

    trained_path = "./trained_chem_vae.pkl"
    with open(trained_path, "wb") as file:
        pkl.dump((trained_path, train_loss, test_loss), file) 



