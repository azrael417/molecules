import time
import torch
import torch.nn as nn
from itertools import chain 
from collections import OrderedDict, namedtuple
from molecules.ml.unsupervised.point_autoencoder.hyperparams import AAE3dHyperparams
from molecules.ml.unsupervised.point_autoencoder.utils import init_weights
from molecules.ml.unsupervised.point_autoencoder.losses import chamfer_loss as cl
from molecules.ml.hyperparams import OptimizerHyperparams, get_optimizer

__all__ = ['AAE3d']

Device = namedtuple('Device', ['encoder', 'generator', 'discriminator'])

class Generator(nn.Module):
    def __init__(self, num_points, num_features, hparams):
        super().__init__()

        # copy some parameters
        self.num_points = num_points
        self.num_features = num_features
        self.z_size = hparams.latent_dim
        self.use_bias = hparams.use_generator_bias
        self.relu_slope = hparams.generator_relu_slope

        # select activation
        if self.relu_slope > 0.:
            self.activation = nn.LeakyReLU(negative_slope = self.relu_slope,
                                           inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
            
        # first layer
        layers = OrderedDict([('linear1', nn.Linear(in_features = self.z_size,
                                                    out_features = hparams.generator_filters[0],
                                                    bias=self.use_bias)),
                              ('relu1', self.activation)])
        
        # intermediate layers
        for idx in range(1, len(hparams.generator_filters)):
            layers.update({'linear{}'.format(idx+1) : nn.Linear(in_features = hparams.generator_filters[idx - 1],
                                                                out_features = hparams.generator_filters[idx],
                                                                bias=self.use_bias)})
            layers.update({'relu{}'.format(idx+1) : self.activation})

        # last layer
        layers.update({'linear{}'.format(idx+2) : nn.Linear(in_features = hparams.generator_filters[-1],
                                                            out_features = self.num_points * (3 + self.num_features),
                                                            bias=self.use_bias)})

        # construct model
        self.model = nn.Sequential(layers)
        
        # init weights
        self.init_weights()
        
        #self.model = nn.Sequential(
        #    nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(in_features=1024, out_features=2048 * 3, bias=self.use_bias),
        #)
        
    def init_weights(self):
        self.model.apply(init_weights)
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, input):
        output = self.model(input.squeeze())
        output = output.view(-1, (3 + self.num_features), self.num_points)
        return output


class Discriminator(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.z_size = hparams.latent_dim
        self.use_bias = hparams.use_discriminator_bias
        self.relu_slope = hparams.discriminator_relu_slope

        # select activation
        if self.relu_slope > 0.:
            self.activation = nn.LeakyReLU(negative_slope = self.relu_slope,
                                           inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        # first layer
        layers = OrderedDict([('linear1', nn.Linear(in_features = self.z_size,
                                                    out_features = hparams.discriminator_filters[0],
                                                    bias = self.use_bias)),
                              ('relu1', self.activation)])

        # intermediate layers
        for idx in range(1, len(hparams.discriminator_filters)):
            layers.update({'linear{}'.format(idx+1) : nn.Linear(in_features = hparams.discriminator_filters[idx - 1],
                                                                out_features = hparams.discriminator_filters[idx],
                                                                bias = self.use_bias)})
            layers.update({'relu{}'.format(idx+1) : self.activation})

        # final layer
        layers.update({'linear{}'.format(idx+2) : nn.Linear(in_features = hparams.discriminator_filters[-1],
                                                            out_features = 1,
                                                            bias = self.use_bias)})

        # construct model
        self.model = nn.Sequential(layers)
        
        # init weights
        self.init_weights()
            
        #self.model = nn.Sequential(
        #
        #    nn.Linear(self.z_size, 512, bias=True),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(512, 512, bias=True),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(512, 128, bias=True),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(128, 64, bias=True),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(64, 1, bias=True)
        #)
        
    def init_weights(self):
        self.model.apply(init_weights)
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        logit = self.model(x)
        return logit


class Encoder(nn.Module):
    def __init__(self, num_points, num_features, hparams):
        super().__init__()

        # copy some parameters
        self.num_points = num_points
        self.num_features = num_features
        self.z_size = hparams.latent_dim
        self.use_bias = hparams.use_encoder_bias
        self.relu_slope = hparams.encoder_relu_slope
        
        # select activation
        if self.relu_slope > 0.:
            self.activation = nn.LeakyReLU(negative_slope = self.relu_slope,
                                           inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        # first layer
        layers = OrderedDict([('conv1', nn.Conv1d(in_channels = (3 + self.num_features),
                                                  out_channels = hparams.encoder_filters[0],
                                                  kernel_size = 1,
                                                  bias = self.use_bias)),
                              ('relu1', self.activation)])

        # intermediate layers
        for idx in range(1, len(hparams.encoder_filters)-1):
            layers.update({'conv{}'.format(idx+1) : nn.Conv1d(in_channels = hparams.encoder_filters[idx - 1],
                                                              out_channels = hparams.encoder_filters[idx],
                                                              kernel_size = 1,
                                                              bias = self.use_bias)})
            layers.update({'relu{}'.format(idx+1) : self.activation})

        # final layer
        layers.update({'conv{}'.format(idx+2) : nn.Conv1d(in_channels = hparams.encoder_filters[-2],
                                                          out_channels = hparams.encoder_filters[-1],
                                                          kernel_size = 1,
                                                          bias = self.use_bias)})

        # construct model
        self.conv = nn.Sequential(layers)

        self.fc = nn.Sequential(
            nn.Linear(hparams.encoder_filters[-1],
                      hparams.encoder_filters[-2],
                      bias=True),
            self.activation
        )

        self.mu_layer = nn.Linear(hparams.encoder_filters[-2], self.z_size, bias=True)
        self.std_layer = nn.Linear(hparams.encoder_filters[-2], self.z_size, bias=True)
        
        # init model
        self.init_weights()
        
        #self.conv = nn.Sequential(
        #    nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1,
        #              bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
        #              bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
        #              bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1,
        #              bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
        #              bias=self.use_bias),
        #)

        #self.fc = nn.Sequential(
        #    nn.Linear(512, 256, bias=True),
        #    nn.ReLU(inplace=True)
        #)

        #self.mu_layer = nn.Linear(256, self.z_size, bias=True)
        #self.std_layer = nn.Linear(256, self.z_size, bias=True)
    
    def init_weights(self):
        self.conv.apply(init_weights)
        self.fc.apply(init_weights)
        self.mu_layer.apply(init_weights)
        self.std_layer.apply(init_weights)
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim = 2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class AAE3dModel(nn.Module):
    def __init__(self, num_points, num_features, hparams, devices):
        super(AAE3dModel, self).__init__()

        # instantiate encoder, generator and discriminator
        self.encoder = Encoder(num_points, num_features, hparams)
        self.generator = Generator(num_points, num_features, hparams)
        self.discriminator = Discriminator(hparams)

        # map to device
        self.encoder = self.encoder.to(devices[0])
        self.generator = self.generator.to(devices[1])
        self.discriminator = self.discriminator.to(devices[2])

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x = self.generator(z)
        return x, mu, logvar
        
    def encode(self, x):
        z, mu, logvar = self.encoder(x)
        return z, mu, logvar

    def generate(self, z):
        x = self.generator(z)
        return x
        
    def discriminate(self, z):
        p = self.discriminator(z)
        return p
    
    def save_weights(self, enc_path, gen_path, disc_path):
        self.encoder.save_weights(enc_path)
        self.generator.save_weights(gen_path)
        self.discriminator.save_weights(disc_path)

    def load_weights(self, enc_path, dec_path):
        self.encoder.load_weights(enc_path)
        self.generator.load_weights(gen_path)
        self.discriminator.load_weights(disc_path)
        
class AAE3d(object):
    """
    Provides high level interface for training, testing and saving VAE
    models. Takes arbitrary encoder/decoder models specified by the choice
    of hyperparameters. Assumes the shape of the data is square.

    Attributes
    ----------
    model : torch.nn.Module (AAE3dModel)
        Underlying Pytorch model with encoder/decoder attributes.

    optimizer : torch.optim.Optimizer
        Pytorch optimizer used to train model.

    loss_func : function
        Loss function used to train model.

    Methods
    -------
    train(train_loader, valid_loader, epochs=1, checkpoint='', callbacks=[])
        Train model

    encode(x)
        Embed data into the latent space.

    generate(embedding)
        Generate matrices from embeddings.
    
    discriminator(embedding)
        Score embedding.

    save_weights(enc_path, gen_path, disc_path)
        Save encoder/generator/discriminator weights.

    load_weights(enc_path, gen_path, disc_path)
        Load saved encoder/generator/discriminator weights.
    """
    
    def __init__(self, num_points, num_features, batch_size,
                     hparams = AAE3dHyperparams(),
                     optimizer_hparams = OptimizerHyperparams(),
                     gpu = None,
                     verbose = True):
        """
        Parameters
        ----------
        num_points : integer
            number of points.

        hparams : molecules.ml.hyperparams.Hyperparams
            Defines the model architecture hyperparameters. Currently implemented
            are AAE3dHyperparams.

        optimizer_hparams : molecules.ml.hyperparams.OptimizerHyperparams
            Defines the optimizer type and corresponding hyperparameters.

        loss: : function, optional
            Defines an optional loss function with inputs (recon_x, x, mu, logvar)
            and ouput torch loss.

        gpu : int, tuple, or None
            Encoder and decoder will train on ...
            If None, cuda GPU device if it is available, otherwise CPU.
            If int, the specified GPU.
            If tuple, the first and second GPUs respectively.

        verbose : bool
            True prints training and validation loss to stdout.
        """
        hparams.validate()
        optimizer_hparams.validate()

        self.verbose = verbose

        # Tuple of encoder, decoder device
        self.devices = Device(*self._configure_device(gpu))

        # model
        self.model = AAE3dModel(num_points, num_features, hparams, self.devices)
        
        # fixed noise vector
        self.num_points = num_points
        self.num_features = num_features
        self.batch_size = batch_size
        self.noise_mu = hparams.noise_mu
        self.noise_std = hparams.noise_std
        self.fixed_noise = torch.FloatTensor(self.batch_size, hparams.latent_dim, 1).to(self.devices[2])
        self.fixed_noise.normal_(mean = self.noise_mu, std = self.noise_std)
        self.noise = torch.FloatTensor(self.batch_size, hparams.latent_dim).to(self.devices[2])

        # TODO: consider making optimizer_hparams a member variable
        # RMSprop with lr=0.001, alpha=0.9, epsilon=1e-08, decay=0.0
        self.optimizer_d = get_optimizer(self.model.discriminator.parameters(), optimizer_hparams)
        self.optimizer_eg = get_optimizer(chain(self.model.encoder.parameters(), self.model.generator.parameters()), 
                                        optimizer_hparams)

        # loss parameters
        self.lambda_gp = hparams.lambda_gp
        self.lambda_rec = hparams.lambda_rec
        self.rec_loss = cl.ChamferLoss()
    
    def _configure_device(self, gpu):
        """
        Configures GPU/CPU device for training AAE. Allows encoder
        and decoder to be trained on seperate devices.

        Parameters
        ----------
        gpu : int, tuple, or None
            Encoder and decoder will train on ...
            If None, cuda GPU device if it is available, otherwise CPU.
            If int, the specified GPU.
            If tuple, the first and second GPUs respectively or None
            option if tuple contains None.

        Returns
        -------
        2-tuple of encoder_device, generator_device
        """

        if (gpu is None) or (isinstance(gpu, tuple) and (None in gpu)):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return device, device, device

        if not torch.cuda.is_available():
            raise ValueError("Specified GPU training but CUDA is not available.")

        if isinstance(gpu, int):
            device = torch.device(f"cuda:{gpu}")
            return device, device, device

        if isinstance(gpu, tuple) and len(gpu) == 3:
            return torch.device(f"cuda:{gpu[0]}"), torch.device(f"cuda:{gpu[1]}"), torch.device(f"cuda:{gpu[2]}")

        raise ValueError("Specified GPU device is invalid. Should be int, 3-tuple or None.")

    def __repr__(self):
        return str(self.model)

    def _loss_fnc_d(self, noise, real_logits, codes, fake_logits):
        # classification loss (critic)
        loss = torch.mean(fake_logits) - torch.mean(real_logits)
        
        # gradient penalty
        alpha = torch.rand(self.batch_size, 1).to(self.devices[2])
        interpolates = noise + alpha * (codes - noise)
        disc_interpolates = self.model.discriminate(interpolates)
        
        gradients = torch.autograd.grad(outputs=disc_interpolates,
                                        inputs=interpolates,
                                        grad_outputs=torch.ones_like(disc_interpolates).to(self.devices[2]),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        gradient_penalty = ((slopes - 1) ** 2).mean()
        loss += self.lambda_gp * gradient_penalty
        
        return loss
        
    def _loss_fnc_eg(self, data, rec_batch, fake_logit):        
        # reconstruction loss
        loss = self.lambda_rec * torch.mean(self.rec_loss(rec_batch, data))

        # add generator loss if requested
        if fake_logit is not None:
            loss += -torch.mean(fake_logit)
        
        return loss

    def train(self, train_loader, valid_loader, epochs=1, checkpoint='', callbacks=[]):
        """
        Train model

        Parameters
        ----------
        train_loader : torch.utils.data.dataloader.DataLoader
            Contains training data

        valid_loader : torch.utils.data.dataloader.DataLoader
            Contains validation data

        epochs : int
            Number of epochs to train for

        checkpoint : str
            Path to checkpoint file to load and resume training
            from the epoch when the checkpoint was saved.

        callbacks : list
            Contains molecules.utils.callback.Callback objects
            which are called during training.
        """
        
        if callbacks:
            logs = {'model': self.model, 'optimizer_d': self.optimizer_d, 'optimizer_eg': self.optimizer_eg}
        else:
            logs = {}

        start_epoch = 1

        if checkpoint:
            start_epoch += self._load_checkpoint(checkpoint)

        for callback in callbacks:
            callback.on_train_begin(logs)
        
        for epoch in range(start_epoch, epochs + 1):

            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

            self._train(train_loader, epoch, callbacks, logs)
            self._validate(valid_loader, callbacks, logs)

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

        for callback in callbacks:
            callback.on_train_end(logs)
        
    def _train(self, train_loader, epoch, callbacks, logs):
        """
        Train for 1 epoch

        Parameters
        ----------
        train_loader : torch.utils.data.dataloader.DataLoader
            Contains training data

        epoch : int
            Current epoch of training

        callbacks : list
            Contains molecules.utils.callback.Callback objects
            which are called during training.

        logs : dict
            Filled with data for callbacks
        """

        self.model.train()
        train_loss_d = 0.
        train_loss_eg = 0.
        for batch_idx, data in enumerate(train_loader):
            
            if self.verbose:
                start = time.time()
            
            if callbacks:
                pass # TODO: add more to logs

            for callback in callbacks:
                callback.on_batch_begin(batch_idx, epoch, logs)

            # copy to gpu
            data = data.to(self.devices[0])
                
            # get reconstruction
            codes, mu, logvar = self.model.encode(data)
            
            # get noise
            self.noise.normal_(mean = self.noise_mu, std = self.noise_std)
            
            # get logits
            real_logits = self.model.discriminate(self.noise)
            fake_logits = self.model.discriminate(codes)
            
            # get loss
            loss_d = self._loss_fnc_d(self.noise, real_logits, codes, fake_logits)
            
            # backward pass
            self.optimizer_d.zero_grad()
            self.model.discriminator.zero_grad()
            loss_d.backward(retain_graph = True)
            
            # optimizer step
            train_loss_d += loss_d.item()
            self.optimizer_d.step()

            # eg step
            rec_batch = self.model.generate(codes)
            fake_logit = self.model.discriminate(codes)
            
            # get loss
            loss_eg = self._loss_fnc_eg(data, rec_batch, fake_logit)
            
            # backward pass
            self.optimizer_eg.zero_grad()
            self.model.generator.zero_grad()
            self.model.encoder.zero_grad()
            loss_eg.backward()
            
            # optimizer step
            train_loss_eg += loss_eg.item()
            self.optimizer_eg.step()

            if callbacks:
                logs['train_loss_d'] = loss_d.item() / len(data)
                logs['train_loss_eg'] = loss_eg.item() / len(data)
                logs['global_step'] = (epoch - 1) * len(train_loader) + batch_idx
            
            if self.verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_d: {:.6f}\tLoss_eg: {:.6f}\tTime: {:.3f}'.format(
                      epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                      100. * (batch_idx + 1) / len(train_loader),
                      loss_d.item() / len(data), loss_eg.item() / len(data), 
                      time.time() - start))

            for callback in callbacks:
                callback.on_batch_end(batch_idx, epoch, logs)

            train_loss_d /= len(train_loader.dataset)
            train_loss_eg /= len(train_loader.dataset)

            if callbacks:
                logs['train_loss_d'] = train_loss_d
                logs['train_loss_eg'] = train_loss_eg
                logs['global_step'] = epoch

            if self.verbose:
                print('====> Epoch: {} Average loss_d: {:.4f} loss_eg: {:.4f}'.format(epoch, train_loss_d, train_loss_eg))
                
    def _validate(self, valid_loader, callbacks, logs):
        """
        Test model on validation set.

        Parameters
        ----------
        valid_loader : torch.utils.data.dataloader.DataLoader
            Contains validation data

        callbacks : list
            Contains molecules.utils.callback.Callback objects
            which are called during training.

        logs : dict
            Filled with data for callbacks
        """
        self.model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data in valid_loader:
                # copy to gpu
                data = data.to(self.devices[0])
                # just reconstruction loss is important here
                recons_batch, mu, logvar = self.model(data)
                valid_loss += self._loss_fnc_eg(data, recons_batch, None).item()

        valid_loss /= len(valid_loader.dataset)

        if callbacks:
            logs['valid_loss'] = valid_loss

        if self.verbose:
            print('====> Validation loss: {:.4f}'.format(valid_loss))
    
    def _load_checkpoint(self, path):
        """
        Loads checkpoint file containing optimizer state and
        encoder/decoder weights.

        Parameters
        ----------
        path : str
            Path to checkpoint file

        Returns
        -------
        Epoch of training corresponding to the saved checkpoint.
        """

        cp = torch.load(path)
        self.model.encoder.load_state_dict(cp['encoder_state_dict'])
        self.model.generator.load_state_dict(cp['generator_state_dict'])
        self.model.discriminator.load_state_dict(cp['discriminator_state_dict'])
        self.optimizer.load_state_dict(cp['optimizer_state_dict'])
        return cp['epoch']
        
    def encode(self, x):
        """
        Embed data into the latent space.

        Parameters
        ----------
        x : torch.Tensor
            Data to encode, could be a batch of data with dimension
            (batch_size, num_points, 3)

        Returns
        -------
        torch.Tensor of embeddings of shape (batch-size, latent_dim)
        """
        return self.model.encode(x)

    def decode(self, embedding):
        """
        Generate matrices from embeddings.

        Parameters
        ----------
        embedding : torch.Tensor
            Embedding data, could be a batch of data with dimension
            (batch-size, latent_dim)

        Returns
        -------
        torch.Tensor of generated matrices of shape (batch-size, num_points, 3)
        """

        return self.model.generate(embedding)
        
    def save_weights(self, enc_path, gen_path, disc_path):
        """
        Save encoder/generator/discriminator weights.

        Parameters
        ----------
        enc_path : str
            Path to save the encoder weights to.

        gen_path : str
            Path to save the generator weights to.
        
        disc_path: str
            Path to save the discriminator weights to.
        """

        self.model.save_weights(enc_path, gen_path, disc_path)

    def load_weights(self, enc_path, gen_path, disc_path):
        """
        Load saved encoder/generator/discriminator weights.

        Parameters
        ----------
        enc_path : str
            Path to load the encoder weights from.

        gen_path : str
            Path to load the generator weights from.
            
        disc_path: str
            Path to load the discriminator weights from.
        """
        self.model.load_weights(enc_path, gen_path, disc_path)