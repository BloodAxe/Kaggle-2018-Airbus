import argparse
import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.common.checkpoint import save_checkpoint, restore_checkpoint
from lib.common.fs import auto_file
from lib.common.random import set_manual_seed
from lib.common.torch_utils import tensor_from_rgb_image
from lib.common.train_utils import log_learning_rate, should_quit, is_better, get_random_name, log_model_graph, \
    compute_total_loss, get_optimizer
from lib.dataset import segmentation as D
from lib.dataset.common import get_train_test_split_for_fold, MASK_KEY, ID_KEY, IMAGE_KEY, get_transform
from lib.losses.segmentation_loss import ShipMaskLoss, ShipEdgeLoss
from lib.metrics.average_meter import AverageMeter
from lib.metrics.jaccard_index import JaccardIndex
from lib.models.models_factory import get_model
from lib.visualizations import visualize_mask_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-dd', '--data-dir', type=str, default='d:\\datasets\\airbus', help='Data dir')
    parser.add_argument('-m', '--model', type=str, default='ternaus_v3', help='')
    parser.add_argument('-b', '--batch-size', type=int, default=10, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='Epoch to run')
    parser.add_argument('-es', '--early-stopping', type=int, default=None,
                        help='Maximum number of epochs without improvement')
    parser.add_argument('-f', '--fold', default=0, type=int, help='Fold to train')
    parser.add_argument('-fe', '--freeze-encoder', type=int, default=0, help='Freeze encoder parameters for N epochs')
    parser.add_argument('-ft', '--fine-tune', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('-lrs', '--lr-scheduler', default=None, help='LR scheduler')
    parser.add_argument('-o', '--optimizer', default='Adam', help='Name of the optimizer')
    parser.add_argument('-p', '--patch-size', type=int, default=768, help='')
    parser.add_argument('-r', '--resume', type=str, default=None, help='Checkpoint filename to resume')
    parser.add_argument('-w', '--workers', default=4, type=int, help='Num workers')
    parser.add_argument('-wd', '--weight-decay', type=float, default=0, help='L2 weight decay')

    args = parser.parse_args()
    set_manual_seed(args.seed)

    train_session_args = vars(args)
    train_session = get_random_name()
    current_time = datetime.now().strftime('%b%d_%H_%M')
    prefix = f'{current_time}_{args.model}_f{args.fold}_{train_session}_{args.patch_size}'

    log_dir = os.path.join('runs', prefix)
    exp_dir = os.path.join('experiments', args.model, prefix)
    os.makedirs(exp_dir, exist_ok=True)

    train_loader, valid_loader = get_dataloaders(args.data_dir, train_batch_size=args.batch_size, fast=args.fast, fold=args.fold, patch_size=args.patch_size)

    # Declare variables we will use during training
    start_epoch = 0
    train_history = pd.DataFrame()

    best_metric_val = 0
    best_lb_checkpoint = os.path.join(exp_dir, f'{prefix}.pth')

    model = get_model(args.model, train_loader.dataset.get_num_classes()).cuda()

    if args.resume:
        fname = auto_file(args.resume)
        start_epoch, train_history, best_score = restore_checkpoint(fname, model)
        print(train_history)
        print('Resuming training from epoch', start_epoch, ' and score', best_score, args.resume)

    writer = SummaryWriter(log_dir)
    writer.add_text('train/params', '```' + json.dumps(train_session_args, indent=2) + '```', 0)
    log_model_graph(writer, model)

    config_fname = os.path.join(exp_dir, f'{train_session}.json')
    with open(config_fname, 'w') as f:
        f.write(json.dumps(train_session_args, indent=2))

    # Main training phase
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_optimizer(args.optimizer, trainable_parameters, args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=50, factor=0.5, min_lr=1e-5)

    train_history, best_metric_val, start_epoch = train(model, optimizer, scheduler, train_loader, valid_loader, writer,
                                                        start_epoch,
                                                        epochs=args.epochs,
                                                        early_stopping=args.early_stopping,
                                                        train_history=train_history,
                                                        experiment_dir=exp_dir,
                                                        best_metric_val=best_metric_val,
                                                        checkpoint_filename=best_lb_checkpoint)

    train_history.to_csv(os.path.join(exp_dir, 'train_history.csv'), index=False)
    print('Training finished')

    # # Make OOF predictions
    # restore_checkpoint(best_lb_checkpoint, model)
    # model.eval()
    #
    # oof_predictions = {}
    # _, valid_ids = get_train_test_split_for_fold(args.fold)
    # for id in valid_ids:
    #     image = read_train_image(id, args.data_dir)
    #     mask = model.predict(image, args.patch_size, args.batch_size)
    #     oof_predictions[id] = to_numpy(mask)
    # np.savez_compressed(os.path.join(exp_dir, f'{args.fold}_oof_predictions.pnz'), **oof_predictions)


def get_dataloaders(data_dir, train_batch_size=1, patch_size=768, workers=4, fold=0, fast=False, oversample_ships=True):
    train_ids, valid_ids = get_train_test_split_for_fold(fold, ships_only=True)
    if fast:
        train_ids = train_ids[:train_batch_size * 8]
        valid_ids = valid_ids[:train_batch_size * 8]

    groundtruth = pd.read_csv(os.path.join(data_dir, 'train_ship_segmentations_v2.csv'))

    trainset = D.SegmentationDataset(sample_ids=train_ids,
                                     data_dir=data_dir,
                                     transform=get_transform(True, patch_size, patch_size),
                                     groundtruth=groundtruth)

    validset = D.SegmentationDataset(sample_ids=valid_ids,
                                     data_dir=data_dir,
                                     transform=get_transform(False, patch_size, patch_size),
                                     groundtruth=groundtruth)

    trainloader = DataLoader(trainset,
                             batch_size=train_batch_size,
                             num_workers=workers,
                             pin_memory=True,
                             drop_last=True,
                             shuffle=True
                             )

    validloader = DataLoader(validset,
                             batch_size=train_batch_size,
                             num_workers=workers,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False,
                             )

    print('Train set', len(trainset), len(trainloader), train_batch_size, 'Valid set', len(validset), len(validloader), train_batch_size)
    return trainloader, validloader


def train(model: nn.Module,
          optimizer: Optimizer,
          scheduler, trainloader: DataLoader,
          validloader: DataLoader,
          writer: SummaryWriter,
          start_epoch: int,
          epochs: int,
          early_stopping,
          train_history: pd.DataFrame,
          experiment_dir: str,
          best_metric_val: float,
          checkpoint_filename: str):
    # Start training loop
    no_improvement_epochs = 0
    epochs_trained = 0

    criterion_weights = None
    criterions = {
        'ship_loss': ShipMaskLoss(),
        'edge_loss': ShipEdgeLoss(),
    }

    target_metric = 'val_mask_iou'
    target_metric_mode = 'max'

    metrics = {
        'mask_iou': JaccardIndex(channel=0),
        'edge_iou': JaccardIndex(channel=1),
    }

    model.zero_grad()
    for epoch in range(start_epoch, start_epoch + epochs):
        # On Epoch begin
        if should_quit(experiment_dir) or (early_stopping is not None and no_improvement_epochs > early_stopping):
            break

        epochs_trained = epoch - start_epoch
        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(epochs_trained)

        log_learning_rate(writer, optimizer, epoch)

        # Epoch
        train_metrics = process_epoch(model, criterions, criterion_weights, metrics, optimizer, trainloader, epoch,
                                      True, writer)
        valid_metrics = process_epoch(model, criterions, criterion_weights, metrics, None, validloader, epoch, False,
                                      writer)

        all_metrics = {}
        all_metrics.update(train_metrics)
        all_metrics.update(valid_metrics)

        # On Epoch End
        summary = {
            'epoch': [int(epoch)],
            'lr': [float(optimizer.param_groups[0]['lr'])]
        }
        for k, v in all_metrics.items():
            summary[k] = [v]

        train_history = train_history.append(pd.DataFrame.from_dict(summary), ignore_index=True)
        print(epoch, summary)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(all_metrics[target_metric], epochs_trained)

        if is_better(all_metrics[target_metric], best_metric_val, target_metric_mode):
            best_metric_val = all_metrics[target_metric]
            save_checkpoint(checkpoint_filename, model, epoch, train_history,
                            metric_name=target_metric,
                            metric_score=best_metric_val)
            print('Checkpoint saved', epoch, best_metric_val, checkpoint_filename)
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        save_checkpoint(os.path.join(experiment_dir, 'last_checkpoint.pth'), model, epoch, train_history,
                        metric_name=target_metric,
                        metric_score=all_metrics[target_metric])

    model.zero_grad()
    return train_history, best_metric_val, epochs_trained + start_epoch + 1


def process_epoch(model,
                  criterions: dict,
                  criterion_weights: Optional[dict],
                  metrics: dict,
                  optimizer,
                  dataloader,
                  epoch: int,
                  is_train,
                  summary_writer,
                  tag=None) -> dict:
    avg_loss = AverageMeter()

    if tag is None:
        tag = 'train' if is_train else 'val'

    epoch_losses = {}

    for key, _ in criterions.items():
        epoch_losses[key] = []

    worst_batch_loss = 0
    worst_batch = None

    best_batch_loss = np.inf
    best_batch = None

    with torch.set_grad_enabled(is_train):
        if is_train:
            model.train()
        else:
            model.eval()

        n_batches = len(dataloader)
        with tqdm(total=n_batches) as tq:
            tq.set_description(f'{tag} epoch %d' % epoch)

            image = None
            true_mask = None
            losses = None
            total_loss = None
            pred_mask = None

            for batch_index, (image, y_true) in enumerate(dataloader):
                batch_size = image.size(0)

                image = image.cuda(non_blocking=True)
                true_mask = y_true[MASK_KEY].cuda(non_blocking=True)

                with torch.autograd.detect_anomaly():
                    if is_train:
                        optimizer.zero_grad()

                    pred_mask = model(image)

                    losses = dict((key, criterions[key](pred_mask, true_mask)) for key in criterions.keys())
                    total_loss = compute_total_loss(losses, criterion_weights)

                    if is_train:
                        total_loss.backward()
                        optimizer.step()

                # Convert mask from logits to probabilities
                pred_mask_sigmoid = pred_mask.detach().sigmoid()

                # Predictions
                total_loss = float(total_loss)
                if total_loss > worst_batch_loss:
                    worst_batch_loss = total_loss
                    worst_batch = {
                        ID_KEY: y_true[ID_KEY],
                        IMAGE_KEY: image.detach().cpu(),
                        'pred_mask': pred_mask_sigmoid.cpu(),
                        'true_mask': true_mask.cpu(),
                    }

                if total_loss < best_batch_loss:
                    best_batch_loss = total_loss
                    best_batch = {
                        ID_KEY: y_true[ID_KEY],
                        IMAGE_KEY: image.detach().cpu(),
                        'pred_mask': pred_mask_sigmoid.cpu(),
                        'true_mask': true_mask.cpu(),
                    }

                # Log losses
                for loss_name in criterions.keys():
                    epoch_losses[loss_name].append(float(losses[loss_name]))

                # Log metrics
                for name, metric in metrics.items():
                    metric.update(pred_mask_sigmoid, true_mask)

                avg_loss.update(total_loss, batch_size)
                tq.set_postfix(loss='{:.3f}'.format(avg_loss.avg))
                tq.update()

            del image, true_mask, losses, total_loss, pred_mask

    for key, metric in metrics.items():
        metric.log_to_tensorboard(summary_writer, f'{tag}/epoch/' + key, epoch)

    # Log losses
    for loss_name, epoch_losses in epoch_losses.items():
        if len(epoch_losses):
            summary_writer.add_scalar(f'{tag}/loss/{loss_name}', np.mean(epoch_losses), epoch)
            summary_writer.add_histogram(f'{tag}/loss/{loss_name}/histogram', np.array(epoch_losses), epoch)

    # Negatives
    negatives = visualize_mask_predictions(worst_batch, show_groundtruth=True)
    for i, image in enumerate(negatives):
        summary_writer.add_image(f'{tag}/negatives/{i}', tensor_from_rgb_image(image), epoch)

    # Positives
    positives = visualize_mask_predictions(best_batch, show_groundtruth=True)
    for i, image in enumerate(positives):
        summary_writer.add_image(f'{tag}/positives/{i}', tensor_from_rgb_image(image), epoch)

    metric_scores = {f'{tag}_loss': avg_loss.avg}
    for key, metric in metrics.items():
        metric_scores[f'{tag}_{key}'] = metric.value()

    return metric_scores


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
