import click
from os.path import join
from torchsummary import summary
from torch.utils.data import DataLoader
from molecules.ml.datasets import ContactMapDataset
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.ml.callbacks import LossCallback, CheckpointCallback, EmbeddingCallback
from molecules.ml.unsupervised.vae import VAE, SymmetricVAEHyperparams, ResnetVAEHyperparams


@click.command()
@click.option('-i', '--input', 'input_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing preprocessed contact matrix data')

@click.option('-o', '--out', 'out_path', required=True,
              type=click.Path(exists=True),
              help='Output directory for model data')

@click.option('-m', '--model_id', required=True,
              help='Model ID in for file naming')

@click.option('-E', '--encoder_gpu', default=None, type=int,
              help='Encoder GPU id')
@click.option('-D', '--decoder_gpu', default=None, type=int,
              help='Decoder GPU id')

@click.option('-e', '--epochs', default=10, type=int,
              help='Number of epochs to train for')

@click.option('-b', '--batch_size', default=128, type=int,
              help='Batch size for training')

@click.option('-t', '--model_type', default='resnet',
              help='Model architecture option: [resnet, symmetric]')

@click.option('-d', '--latent_dim', default=10, type=int,
              help='Number of dimensions in latent space')

def main(input_path, out_path, model_id, encoder_gpu,
         decoder_gpu, epochs, batch_size, model_type, latent_dim):
    """Example for training Fs-peptide with either Symmetric or Resnet VAE."""

    assert model_type in ['symmetric', 'resnet']

    # Note: See SymmetricVAEHyperparams, ResnetVAEHyperparams class definitions
    #       for hyperparameter options. 

    if model_type == 'symmetric':
        # Optimal Fs-peptide params
        #fs_peptide_hparams ={'filters': [100, 100, 100, 100],
        #                     'kernels': [5, 5, 5, 5],
        #                     'strides': [1, 2, 1, 1],
        #                     'affine_widths': [64],
        #                     'affine_dropouts': [0],
         #                    'latent_dim': latent_dim}
        hparams = {'filters': [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
           'kernels': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
           'strides': [1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
           'affine_widths': [128],
           'affine_dropouts': [0],
           'latent_dim': 20}


        input_shape = (1, 3768, 3768)
        squeeze = False
        hparams = SymmetricVAEHyperparams(**hparams)

        #hparams = SymmetricVAEHyperparams(**fs_peptide_hparams)

    elif model_type == 'resnet':
        input_shape = (22, 22)
        squeeze = True # Specify no ones in training data shape
        hparams = ResnetVAEHyperparams(input_shape, latent_dim=11)

    optimizer_hparams = OptimizerHyperparams(name='RMSprop', hparams={'lr':0.00001})

    vae = VAE(input_shape, hparams, optimizer_hparams,
              gpu=(encoder_gpu, decoder_gpu))

    # Diplay model
    print(vae)
    summary(vae.model, input_shape)

    # Load training and validation data
    train_loader = DataLoader(ContactMapDataset(input_path, split='train', squeeze=squeeze),
                              batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(ContactMapDataset(input_path, split='valid', squeeze=squeeze),
                              batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # For ease of training multiple models
    model_path = join(out_path, f'model-{model_id}')

    # Optional callbacks
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    loss_callback = LossCallback(join(model_path, 'loss.json'), writer)
    checkpoint_callback = CheckpointCallback(out_dir=join(model_path, 'checkpoint'))
    #embedding_callback = EmbeddingCallback(input_path,
    #                                       out_dir=join(model_path, 'embedddings'),
    #                                       squeeze=squeeze,
    #                                       writer=writer)

    # Train model with callbacks
    vae.train(train_loader, valid_loader, epochs,
              callbacks=[loss_callback, checkpoint_callback])#, embedding_callback])

    # Save loss history to disk.
    loss_callback.save(join(model_path, 'loss.json'))

    # Save hparams to disk
    hparams.save(join(model_path, 'model-hparams.pkl'))
    optimizer_hparams.save(join(model_path, 'optimizer-hparams.pkl'))

    # Save final model weights to disk
    vae.save_weights(join(model_path, 'encoder-weights.pt'),
                     join(model_path, 'decoder-weights.pt'))

    # Output directory structure
    #  out_path
    # ├── model_path
    # │   ├── checkpoint
    # │   │   ├── epoch-1-20200606-125334.pt
    # │   │   └── epoch-2-20200606-125338.pt
    # │   ├── decoder-weights.pt
    # │   ├── encoder-weights.pt
    # │   ├── loss.json
    # │   ├── model-hparams.pkl
    # │   └── optimizer-hparams.pkl

if __name__ == '__main__':
    main()
