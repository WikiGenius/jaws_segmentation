import os
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import wandb

import glob
from tqdm import tqdm
import logging
from pathlib import Path

# helper files
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
from utils.load_dataset import JawsDataset


def train_net(net,
              train_loaders,
              validation_loaders,
              n_train_dict,
              n_validation_dict,
              device,
              plane,
              experiment,
              img_size,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              amp: bool = False):
    '''
    plane: str = 'axial' | 'coronal' | sagittal
    '''
    dir_checkpoint = Path(f'./checkpoints/{plane}')

    experiment.config.update(dict(epochs=epochs, plane=plane, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_size=img_size,
                                  amp=amp))

    train_loader = train_loaders[plane]
    val_loader = validation_loaders[plane]
    n_train = n_train_dict[plane]
    n_val = n_validation_dict[plane]
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        plane:     {plane}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images size:  {img_size}
        Mixed Precision (amp): {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        batch_step = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                batch_step += 1
                images, true_masks = batch['image'], batch['mask']
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                true_masks = true_masks.squeeze()
                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                    F.one_hot(true_masks, net.n_classes).permute(
                            0, 3, 1, 2).float(),
                        multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' +
                                       tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' +
                                       tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info(
                            'Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        logging.info(f'''
            Avg. epoch train loss":    {epoch_loss / batch_step}
                       ''')
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint) +
                       '/checkpoint_epoch{}.pth'.format(epoch))
            logging.info(f'Checkpoint {epoch} saved!')


def train_plane(plane, experiment, img_size, n_classes=3, train=True, load_last_model=False, load_interrupted_model=True, bilinear=False, epochs=1, batch_size=16, learning_rate=1e-5, val_percent=0.1, amp=False):
    '''
    plane: str = 'axial' | 'coronal' | sagittal
    '''
    if train:
        try:
            dir_checkpoint = Path(f'./checkpoints/{plane}')
            checkpoint = sorted(os.listdir(
                str(Path(f'./checkpoints/{plane}'))))[-1]
            if load_last_model:
                load_model = str(dir_checkpoint) + f'/{checkpoint}'
            elif load_interrupted_model:
                load_model = str(dir_checkpoint) + f'/INTERRUPTED.pth'
            else:
                load_model = False
        except:
            load_model = False
        logging.basicConfig(level=logging.INFO,
                            format='%(levelname)s: %(message)s')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')

        # n_channels=1 for gray images
        # n_classes is the number of probabilities you want to get per pixel
        net = UNet(n_channels=1, n_classes=n_classes, bilinear=bilinear)

        logging.info(f'Network:\n'
                     f'\t{net.n_channels} input channels\n'
                     f'\t{net.n_classes} output channels (classes)\n'
                     f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

        if load_model:
            net.load_state_dict(torch.load(load_model, map_location=device))
            logging.info(f'Model loaded from {load_model}')
        net.to(device=device)
        try:
            train_net(net=net,
                      plane=plane,
                      experiment=experiment,
                      img_size=img_size,
                      epochs=epochs,
                      batch_size=batch_size,
                      learning_rate=learning_rate,
                      device=device,
                      val_percent=val_percent,
                      amp=amp,
                      )
        except KeyboardInterrupt:
            torch.save(net.state_dict(), str(
                dir_checkpoint)+'/INTERRUPTED.pth')
            logging.info('Saved interrupt')
            raise
    else:
        print("No need for training")
