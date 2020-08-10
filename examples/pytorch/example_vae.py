import os
import click
from os.path import join
from torchsummary import summary
from torch.utils.data import DataLoader, Subset
from molecules.ml.datasets import ContactMapDataset
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.ml.callbacks import LossCallback, CheckpointCallback, EmbeddingCallback
from molecules.ml.unsupervised.vae import VAE, SymmetricVAEHyperparams, ResnetVAEHyperparams
import torch.distributed as dist
import torch.nn.parallel.DistributedDataParallel as DDP

@click.command()
@click.option('-i', '--input', 'input_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing preprocessed contact matrix data')

@click.option('-o', '--out', 'out_path', required=True,
              type=click.Path(exists=True),
              help='Output directory for model data')

@click.option('-m', '--model_id', required=True,
              help='Model ID in for file naming')

@click.option('-h', '--dim1', required=True, type=int,
              help='H of (H,W) shaped contact matrix')

@click.option('-w', '--dim2', required=True, type=int,
              help='W of (H,W) shaped contact matrix')

@click.option('-s', '--sparse', is_flag=True,
              help='Specifiy whether input matrices are sparse format')

@click.option('-a', '--amp', is_flag=True,
              help='Specifiy to enable AMP')

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

@click.option('-wp', '--wandb_project_name', default=None, type=str,
              help='Project name for wandb logging')

def main(input_path, out_path, model_id, dim1, dim2, encoder_gpu, sparse,
         amp, decoder_gpu, epochs, batch_size, model_type, latent_dim,
         wandb_project_name):
    """Example for training Fs-peptide with either Symmetric or Resnet VAE."""

    # init distributed
    dist.init_process_group(backend='nccl', init_method='env://')
    comm_size = torch.distributed.get_world_size()
    comm_rank = torch.distributed.get_rank()
    comm_ngpu = torch.cuda.device_count()
    if encoder_gpu == decoder_gpu:
        comm_local_rank = comm_rank % comm_ngpu
    else:
        comm_local_rank = comm_rank % (comm_ngpu // 2)

    assert model_type in ['symmetric', 'resnet']

    # Note: See SymmetricVAEHyperparams, ResnetVAEHyperparams class definitions
    #       for hyperparameter options. 

    if model_type == 'symmetric':
        # Optimal Fs-peptide params
        fs_peptide_hparams ={'filters': [100, 100, 100, 100],
                             'kernels': [5, 5, 5, 5],
                             'strides': [1, 2, 1, 1],
                             'affine_widths': [64],
                             'affine_dropouts': [0],
                             'latent_dim': latent_dim}

        input_shape = (1, dim1, dim2)
        hparams = SymmetricVAEHyperparams(**fs_peptide_hparams)

    elif model_type == 'resnet':

        resnet_hparams = {'max_len': dim1,
                          'nchars': dim2,
                          'latent_dim': latent_dim,
                          'dec_filters': dim1,
                          'output_activation': 'None'}

        input_shape = (dim1, dim1)
        hparams = ResnetVAEHyperparams(**resnet_hparams)

    optimizer_hparams = OptimizerHyperparams(name='RMSprop', hparams={'lr':0.00001})

    # place encoder and decoder
    vae = VAE(input_shape, hparams, optimizer_hparams,
              gpu = (encoder_gpu, decoder_gpu), enable_amp = amp)

    # do we want distributed training?
    if comm_size > 1:
        vae.model = DDP(vae.model, device_ids = None, output_device = None)
    
    # Diplay model
    print(vae)
    # Only print summary when encoder_gpu is None or 0
    summary(vae.model, input_shape)

    # Load training and validation data
    # training: chunk the dataset
    train_dataset = ContactMapDataset(input_path,
                                      "contact_maps",
                                      "rmsd",
                                      input_shape,
                                      split='train',
                                      sparse=sparse)
    if comm_size > 1:
        chunksize = len(train_dataset) // comm_size
        train_dataset = subset(train_dataset, range(comm_rank * chunksize, (comm_rank + 1) * chunksize))
    
    train_loader = DataLoader(train_dataset,
                              batch_size = batch_size,
                              shuffle = True,
                              pin_memory = True)

    # validation: do not chunk the dataset
    valid_dataset = ContactMapDataset(input_path,
                                      "contact_maps",
                                      "rmsd",
                                      input_shape,
                                      split='valid',
                                      sparse=sparse)
    
    valid_loader = DataLoader(valid_dataset,
                              batch_size = batch_size,
                              shuffle = True,
                              pin_memory = True)

    # For ease of training multiple models
    model_path = join(out_path, f'model-{model_id}')

    # do we want wandb
    wandb_config = None
    if (wandb_project_name is not None) and (comm_rank == 0):
        import wandb
        wandb.init(project = wandb_project_name,
                   name = model_id,
                   id = model_id,
                   resume = False)
        wandb_config = wandb.config
        
        # log HP
        wandb_config.dim1 = dim1
        wandb_config.dim2 = dim2
        wandb_config.latent_dim = latent_dim
        
        # optimizer
        wandb_config.optimizer_name = optimizer_hparams.name
        for param in optimizer_hparams.hparams:
            wandb_config["optimizer_" + param] = optimizer_hparams.hparams[param]
            
        # watch model
        wandb.watch(vae.model)
    
    # Optional callbacks
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    loss_callback = LossCallback(join(model_path, 'loss.json'), writer, wandb_config)

    if comm_rank == 0:
        checkpoint_callback = CheckpointCallback(out_dir=join(model_path, 'checkpoint'))
        embedding_callback = EmbeddingCallback(out_dir = join(model_path, 'embedddings'),
                                               path = input_path,
                                               rmsd_name = "rmsd",
                                               projection_type = "3d_project",
                                               sample_interval = len(valid_dataset) // 1000,
                                               writer = writer,
                                               wandb_config = wandb_config)

    # Train model with callbacks
    if comm_rank == 0:
        callbacks = [loss_callback, checkpoint_callback, embedding_callback]
    else:
        callbacks = [loss_callback]

    # train the model
    vae.train(train_loader, valid_loader, epochs, callbacks = callbacks)

    # Save stuff
    if comm_rank == 0:
        # Save loss history to disk.
        loss_callback.save(join(model_path, 'loss.json'))

        # Save hparams to disk
        hparams.save(join(model_path, 'model-hparams.json'))
        optimizer_hparams.save(join(model_path, 'optimizer-hparams.json'))

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
