import os
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import re

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
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class Net_Utility:
    def __init__(self, plane: str, train_loaders, validation_loaders, n_train_dict, n_validation_dict, img_size, n_classes=3, save_checkpoint=True, train=True, load_last_model=False, load_interrupted_model=True, bilinear=False, epochs=1, batch_size=16, learning_rate=1e-5, val_percent=0.1, amp=False):

        self.plane = plane
        self.train_loaders = train_loaders
        self.validation_loaders = validation_loaders
        self.n_train_dict = n_train_dict
        self.n_validation_dict = n_validation_dict
        self.img_size = img_size
        self.n_classes = n_classes
        self.save_checkpoint = save_checkpoint
        self.train = train
        self.load_last_model = load_last_model
        self.load_interrupted_model = load_interrupted_model
        self.bilinear = bilinear
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_percent = val_percent
        self.amp = amp
        self.train_loss = []
        self.validation_loss = []
        self.train_score = []
        self.validation_score = []
        self.learning_rates = []
        self.current_epoch = 0
        self.global_step = 0
        self.results = dict()
        pass

    def train_plane(self):
        '''
        plane: str = 'axial' | 'coronal' | sagittal
        '''
        if self.train:
            try:

                ######
                check_folders = sorted(os.listdir('checkpoints'))
                nums = sorted([int(re.search(r'[0-9]+', folder).group())
                              for folder in check_folders])
                last_num_check_folders = nums[-1]

                dir_checkpoint = Path(
                    f'./checkpoints/v{last_num_check_folders}/{self.plane}')

                check_vn_epochs = sorted(os.listdir(dir_checkpoint))
                nums = sorted([int(re.search(r'[0-9]+', folder).group())
                              for folder in check_vn_epochs])
                last_num_check_vn_epochs = nums[-1]
                checkpoint = Path(
                    f'{dir_checkpoint}/checkpoint_epoch{last_num_check_vn_epochs}.pth')

                if self.load_last_model:
                    load_model = str(dir_checkpoint) + f'/{checkpoint}'
                elif self.load_interrupted_model:
                    load_model = str(dir_checkpoint) + f'/INTERRUPTED.pth'
                else:
                    load_model = False
            except:
                load_model = False
            logging.basicConfig(level=logging.INFO,
                                format='%(levelname)s: %(message)s')
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f'Using device {device}')

            # n_channels=1 for gray images
            # n_classes is the number of probabilities you want to get per pixel
            net = UNet(n_channels=1, n_classes=self.n_classes,
                       bilinear=self.bilinear)

            logging.info(f'Network:\n'
                         f'\t{net.n_channels} input channels\n'
                         f'\t{net.n_classes} output channels (classes)\n'
                         f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

            if load_model:
                net.load_state_dict(torch.load(
                    load_model, map_location=device))
                logging.info(f'Model loaded from {load_model}')
            net.to(device=device)
            try:
                self._train_net(net, device)
            except KeyboardInterrupt:
                torch.save(net.state_dict(), str(
                    dir_checkpoint)+'/INTERRUPTED.pth')
                logging.info('Saved interrupt')
                raise
        else:
            print("No need for training")

    def display_results(self):

        results = self.get_results()
        train_loss = results['loss']['train']

        validation_score = results['score']['validation']
        validation_score = [s.item()for s in validation_score]

        learning_rates = results['learning_rate']

        global_steps = results['global_step']

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"colspan": 2}, None],
                   [{}, {}]],
            subplot_titles=("Train loss", "validation_score", "learning_rates"))

        fig.add_trace(go.Scatter(name='Train', x=list(range(1, global_steps + 1)), y=train_loss),
                      row=1, col=1)

        fig.add_trace(go.Scatter(name='Validation', x=list(range(1, len(validation_score) + 1)), y=validation_score),
                      row=2, col=1)
        fig.add_trace(go.Scatter(name='lr', x=list(range(1, len(learning_rates) + 1)), y=learning_rates),
                      row=2, col=2)
        fig.update_layout(height=400, width=600, showlegend=False,
                          title_text=f"Results plane {self.plane}")
        fig.show()

    def get_results(self):

        self.results['loss'] = {"train": self.train_loss,
                                "validation": self.validation_loss}
        self.results['score'] = {
            "train": self.train_score, "validation": self.validation_score}
        self.results['learning_rate'] = self.learning_rates
        self.results['current_epoch'] = self.current_epoch
        self.results['global_step'] = self.global_step

        return self.results

    def _train_net(self, net, device):
        '''
        plane: str = 'axial' | 'coronal' | sagittal
        '''

        check_folders = sorted(os.listdir('checkpoints'))
        nums = sorted([int(re.search(r'[0-9]+', folder).group())
                      for folder in check_folders])
        last_num = nums[-1]
        if os.path.isdir(f'./checkpoints/v{last_num}/{self.plane}'):
            last_num += 1

        dir_checkpoint = Path(f'./checkpoints/v{last_num}/{self.plane}')
        # (Initialize logging)
        experiment = wandb.init(
            project=f'jaws_segmentation_{self.plane}', entity='muhammed-elyamani')
        experiment.config.update(dict(epochs=self.epochs, plane=self.plane, batch_size=self.batch_size, learning_rate=self.learning_rate,
                                      val_percent=self.val_percent, save_checkpoint=self.save_checkpoint, img_size=self.img_size,
                                      amp=self.amp))

        train_loader = self.train_loaders[self.plane]
        val_loader = self.validation_loaders[self.plane]
        n_train = self.n_train_dict[self.plane]
        n_val = self.n_validation_dict[self.plane]
        logging.info(f'''Starting training:
            Epochs:          {self.epochs}
            plane:     {self.plane}
            Batch size:      {self.batch_size}
            Learning rate:   {self.learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {self.save_checkpoint}
            Device:          {device.type}
            Images size:  {self.img_size}
            Mixed Precision (amp): {self.amp}
        ''')

        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer = optim.RMSprop(
            net.parameters(), lr=self.learning_rate, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', patience=2)  # goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        criterion = nn.CrossEntropyLoss()
        self.global_step = 0

        # 5. Begin training
        for epoch in range(1, self.epochs+1):
            net.train()
            epoch_loss = 0
            batch_step = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{self.epochs}', unit='img') as pbar:
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
                    with torch.cuda.amp.autocast(enabled=self.amp):
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
                    self.global_step += 1
                    epoch_loss += loss.item()
                    self.train_loss.append(loss.item())
                    self.current_epoch = epoch
                    experiment.log({
                        'train loss': loss.item(),
                        'step': self.global_step,
                        'epoch': epoch
                    })
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Evaluation round
                    division_step = (n_train // (10 * self.batch_size))
                    if division_step > 0:
                        if self.global_step % division_step == 0:
                            histograms = {}
                            for tag, value in net.named_parameters():
                                tag = tag.replace('/', '.')
                                histograms['Weights/' +
                                           tag] = wandb.Histogram(value.data.cpu())
                                histograms['Gradients/' +
                                           tag] = wandb.Histogram(value.grad.data.cpu())

                            val_score = evaluate(net, val_loader, device)
                            scheduler.step(val_score)
                            self.validation_score.append(val_score)
                            self.learning_rates.append(
                                optimizer.param_groups[0]['lr'])
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
                                'step': self.global_step,
                                'epoch': epoch,
                                **histograms
                            })

            logging.info(f'''
                Avg. epoch train loss":    {epoch_loss / batch_step}
                           ''')
            if self.save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint) +
                           '/checkpoint_epoch{}.pth'.format(epoch))
                logging.info(f'Checkpoint {epoch} saved!')
