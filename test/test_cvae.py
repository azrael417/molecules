import os
import pytest
import numpy as np
import torch
from torchsummary import summary
from molecules.utils import open_h5
from torch.utils.data import Dataset, DataLoader
from molecules.ml.unsupervised.vae import SymmetricVAEHyperparams
from molecules.ml.unsupervised.vae import VAE
from molecules.ml.hyperparams import OptimizerHyperparams

class TestVAE:

    class DummyContactMap(Dataset):
            def __init__(self, input_shape, size=200):
                self.maps = np.random.normal(size=(size, *input_shape))

            def __len__(self):
                return len(self.maps)

            def __getitem__(self, idx):
                return torch.from_numpy(self.maps[idx]).to(torch.float32)


    class SparseSquareContactMap(Dataset):
            def __init__(self, input_shape, size=200):
                self.input_shape = input_shape
                self.maps = np.array([np.eye(input_shape[1]) for _ in range(size)])
                # Creates size identity matrices. Total shape: (size, input_shape)

            def __len__(self):
                return len(self.maps)

            def __getitem__(self, idx):
                from scipy.sparse import coo_matrix
                coo = coo_matrix(self.maps[idx])

                values = coo.data
                indices = np.vstack((coo.row, coo.col))
                i = torch.LongTensor(indices)
                v = torch.FloatTensor(values)

                return torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))


    # TODO: find elegant way to get the input_shape for the model initialization
    class ContactMap(Dataset):
        def __init__(self, path, split_ptc=0.8, split='train', squeeze=False):
            with open_h5(path) as input_file:
                # Access contact matrix data from h5 file
                data = np.array(input_file['contact_maps'])

            # 80-20 train validation split index
            split_ind = int(split_ptc * len(data))

            if split == 'train':
                self.data = data[:split_ind]
            elif split == 'valid':
                self.data = data[split_ind:]
            else:
                raise ValueError(f'Parameter split={split} is invalid.')

            # TODO: in future contact map Dataset, pass in device to precompute
            #       the operation


            # TODO: this reshape code may not be the best solution. revisit
            num_residues = self.data.shape[2]
            assert num_residues == 22

            if squeeze:
                shape = (-1, num_residues, num_residues)
            else:
                shape = (-1, 1, num_residues, num_residues)

            self.data = torch.from_numpy(self.data.reshape(shape)).to(torch.float32)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]


    @classmethod
    def setup_class(self):
        self.epochs = 1
        self.batch_size = 100
        self.input_shape = (1, 22, 22) # Use FSPeptide sized contact maps
        self.train_loader = DataLoader(TestVAE.DummyContactMap(self.input_shape),
                                       batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(TestVAE.DummyContactMap(self.input_shape),
                                      batch_size=self.batch_size, shuffle=True)


        # Optimal Fs-peptide params
        fs_peptide_hparams = {'filters': [100, 100, 100, 100],
                              'kernels': [5, 5, 5, 5],
                              'strides': [1, 2, 1, 1],
                              'affine_widths': [64],
                              'affine_dropouts': [0],
                              'latent_dim': 10}

        diff_filters_hparams = {'filters': [100, 64, 64, 100],
                                'kernels': [5, 5, 3, 7],
                                'strides': [1, 2, 1, 2],
                                'affine_widths': [64],
                                'affine_dropouts': [0],
                                'latent_dim': 10}

        strided_hparams = {'filters': [100, 100, 100, 100],
                           'kernels': [5, 5, 5, 5],
                           'strides': [1, 2, 2, 1],
                           'affine_widths': [64],
                           'affine_dropouts': [0],
                           'latent_dim': 10}

        hparams = {'filters': [64, 64, 64, 64],
                   'kernels': [3, 3, 3, 3],
                   'strides': [1, 2, 1, 1],
                   'affine_widths': [128],
                   'affine_dropouts': [0],
                   'latent_dim': 3}

        self.hparams = SymmetricVAEHyperparams(**fs_peptide_hparams)
        self.optimizer_hparams = OptimizerHyperparams(name='RMSprop', hparams={'lr':0.00001})

        # For testing saving and loading weights
        self.enc_path = os.path.join('.', 'test', 'data', 'encoder-weights.pt')
        self.dec_path = os.path.join('.', 'test', 'data', 'decoder-weights.pt')

    def test_padding(self):
        from molecules.ml.unsupervised.vae.utils import same_padding


        # Square input_dim and padding
        input_dim = (22, 22)
        kernel_size = 3

        assert same_padding(input_dim, kernel_size, stride=1) == (1, 1) # Stride 1
        assert same_padding(input_dim, kernel_size, stride=2) == (1, 1) # Test fs-peptide
        assert same_padding(input_dim, 5, stride=1) == (2, 2) # Optimal fs-peptide
        assert same_padding(input_dim, 5, stride=2) == (1, 1) # Optimal fs-peptide
        assert same_padding((75, 75), 2, 2) == (1, 1) # Resnet Autoencoder

        assert same_padding((5, 5), 2, 2) == (1, 1)

        # Rectangular input_dim and padding
        assert same_padding((22, 10), kernel_size, stride=1) == (1, 1)
        assert same_padding((22, 15), kernel_size, stride=1) == (1, 1)
        assert same_padding((22, 3), kernel_size, stride=1) == (1, 1)

        assert same_padding((75, 5), 2, stride=2) == (1, 1)



    def test_conv_output_shape(self):
        from molecules.ml.unsupervised.vae.utils import conv_output_shape

        # Optimal fs-peptide
        assert conv_output_shape(input_dim=(22, 22), kernel_size=5, stride=1, padding=(2, 2),
                                 num_filters=100) == (100, 22, 22)
        assert conv_output_shape(input_dim=(22, 22), kernel_size=5, stride=2, padding=(1, 1),
                                 num_filters=100) == (100, 10, 10)
        # Test fs-peptide
        assert conv_output_shape(input_dim=(22, 22), kernel_size=3, stride=1, padding=(1, 1),
                                 num_filters=64) == (64, 22, 22)
        assert conv_output_shape(input_dim=(22, 22), kernel_size=3, stride=2, padding=(1, 1),
                                 num_filters=64) == (64, 11, 11)

        # TODO: fix
        #assert conv_output_shape(input_dim=(10, 10), kernel_size=5, stride=2,
        #                         num_filters=100, padding=(1,1), transpose=True) == (100, 22, 22)

    def _test_cvae_real_data(self):

        path = './test/cvae_input.h5'

        train_loader = DataLoader(TestVAE.ContactMap(path, split='train'),
                                  batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TestVAE.ContactMap(path, split='valid'),
                                 batch_size=self.batch_size, shuffle=True)

        vae = VAE(self.input_shape, self.hparams, self.optimizer_hparams)

        print(vae)
        summary(vae.model, self.input_shape)

        vae.train(train_loader, test_loader, self.epochs)

    def _test_rectangular_data_symmetric_vae(self):

        rectangular_shape = (1, 22, 22)
        #rectangular_shape = (1, 24, 524)

        train_loader = DataLoader(TestVAE.DummyContactMap(rectangular_shape),
                                  batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TestVAE.DummyContactMap(rectangular_shape),
                                 batch_size=self.batch_size, shuffle=True)

        vae = VAE(rectangular_shape, self.hparams, self.optimizer_hparams)

        print(vae)
        summary(vae.model, rectangular_shape)

        vae.train(train_loader, test_loader, self.epochs)

    def test_sparse_data_symmetric_vae(self):

        input_shape = (22, 22)

        train_loader = DataLoader(TestVAE.SparseSquareContactMap(input_shape),
                                  batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TestVAE.SparseSquareContactMap(input_shape),
                                 batch_size=self.batch_size, shuffle=True)

        vae = VAE(input_shape, self.hparams, self.optimizer_hparams)

        print(vae)
        summary(vae.model, input_shape)

        vae.train(train_loader, test_loader, self.epochs)

    def _test_rectangular_data_resnet_vae(self):

        #max_len, nchars = 24, 524

        # (22, 22) input use latent_dim=11, dec_filters=22
        max_len, nchars = 22, 22


        rectangular_shape = (max_len, nchars)

        train_loader = DataLoader(TestVAE.DummyContactMap(rectangular_shape),
                                  batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TestVAE.DummyContactMap(rectangular_shape),
                                 batch_size=self.batch_size, shuffle=True)

        from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams
        hparams = ResnetVAEHyperparams(max_len=max_len, nchars=nchars, latent_dim=11,
                                       dec_filters=22)

        vae = VAE(rectangular_shape, hparams, self.optimizer_hparams)

        print(vae)
        summary(vae.model, rectangular_shape)

        vae.train(train_loader, test_loader, self.epochs)

    def _test_encode_decode(self):
        vae = VAE(self.input_shape, self.hparams, self.optimizer_hparams)
        vae.train(self.train_loader, self.test_loader, self.epochs)

        test_data = TestVAE.DummyContactMap(self.input_shape, 100)[:]

        embeddings = vae.encode(test_data)

        recons = vae.decode(embeddings)

    def _test_save_load_weights(self):
        vae1 = VAE(self.input_shape, self.hparams, self.optimizer_hparams)
        vae1.train(self.train_loader, self.test_loader, self.epochs)
        vae1.save_weights(self.enc_path, self.dec_path)

        vae2 = VAE(self.input_shape, self.hparams, self.optimizer_hparams)
        vae2.load_weights(self.enc_path, self.dec_path)

        # Checks that all model weights are exactly equal
        for va1_params, vae2_params in zip(vae1.model.state_dict().values(),
                                           vae2.model.state_dict().values()):
            assert torch.equal(va1_params, va1_params)


        # Checks that weights can be loaded into encoder/decoder modules seperately
        from molecules.ml.unsupervised.vae.symmetric import (SymmetricEncoderConv2d,
                                                             SymmetricDecoderConv2d)

        encoder = SymmetricEncoderConv2d(self.input_shape, self.hparams)
        encoder.load_weights(self.enc_path)

        decoder = SymmetricDecoderConv2d(self.input_shape, self.hparams,
                                         encoder.shape)
        decoder.load_weights(self.dec_path)

    def _test_resnet_vae(self):
        from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams

        input_shape = (1200, 1200)
        hparams = ResnetVAEHyperparams(input_shape, latent_dim=150)

        vae = VAE(input_shape, hparams, self.optimizer_hparams)

        print(vae)
        summary(vae.model, input_shape)

    def _test_resnet_vae_training(self):
        from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams

        path = './test/cvae_input.h5'

        train_loader = DataLoader(TestVAE.ContactMap(path, split='train', squeeze=True),
                                  batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TestVAE.ContactMap(path, split='valid', squeeze=True),
                                 batch_size=self.batch_size, shuffle=True)

        input_shape = (22, 22)
        hparams = ResnetVAEHyperparams(input_shape, latent_dim=11)

        vae = VAE(input_shape, hparams, self.optimizer_hparams)

        print(vae)
        summary(vae.model, input_shape)

        vae.train(train_loader, test_loader, self.epochs)

    def _test_residual_module(self):
        from molecules.ml.unsupervised.vae.resnet.residual_module import ResidualConv1d

        input_shape, filters, kernel_size, activation = (1, 22*22), 10, 3, 'ReLU'

        res = ResidualConv1d(input_shape, filters, kernel_size, activation)

        print(res)
        summary(res, input_shape)

        for x in self.train_loader:
            # Flatten contact matrix
            x = x.view(-1, *input_shape)
            out = res(x)

        # Test with shrink=True
        res = ResidualConv1d(input_shape, filters,
                             kernel_size, activation,
                             shrink=True)

        print(res)
        summary(res, input_shape)

        for x in self.train_loader:
            # Flatten contact matrix
            x = x.view(-1, *input_shape)
            out = res(x)

    def test_resnet_big_input(self):
        big_shape = (3768, 3768) # GB: 0.113582592

        train_loader = DataLoader(TestVAE.DummyContactMap(big_shape, size=5),
                                  batch_size=2, shuffle=True)
        test_loader = DataLoader(TestVAE.DummyContactMap(big_shape, size=5),
                                 batch_size=5, shuffle=True)


        from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams
        hparams = ResnetVAEHyperparams(nchars=3768, max_len=3768, latent_dim=24)

        vae = VAE(big_shape, hparams, self.optimizer_hparams)

        print(vae)
        summary(vae.model, big_shape)

        vae.train(train_loader, test_loader, self.epochs)

    @classmethod
    def teardown_class(self):
        # Delete file to clean testing directories
        # os.remove(self.enc_path)
        # os.remove(self.dec_path)
        pass
