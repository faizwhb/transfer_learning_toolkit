import argparse
from torch.utils.tensorboard import SummaryWriter
import logging
import torch
import torch.nn as nn
import numpy as np


import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import config_utils.config_eval as config_eval
from tools.datasets.transformers import make_transform
from tools.datasets.csv_dataset import Dataset_from_CSV
from tools.models import model_getter
from tools.evaluation_metrics.precision_recall_scores import compute_average_precision_score, compute_average_recall


def parse_args():
    parser = argparse.ArgumentParser(description='Simple Classification Baseline on PyTorch Training')
    parser.add_argument('--config_name', default='config/train.json')
    args = parser.parse_args()
    return args


def my_collate(batch):
    # collate function for skipping corrupted or broken files
    batch = list(filter (lambda x:x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def evaluate(test_loader, model, device, num_classes):
    # switch to evaluation mode
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    targets = []
    predictions = []
    with torch.no_grad():
        for i, (input, target, index) in enumerate(test_loader):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # accumulate labels to compute evaluation metrics
            targets += np.squeeze(target.tolist()).tolist()
            # compute output
            _, outputs = model(input)
            _, predicted = torch.max(outputs, dim=1)

            # accumulate labels to compute evaluation metrics
            predictions += np.squeeze(predicted.tolist()).tolist()
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100*correct/total
    average_precision = compute_average_precision_score(ground_truth=targets,
                                                        predictions=predictions, num_classes=num_classes)
    average_recall = compute_average_recall(ground_truth=targets, predictions=predictions,
                                            num_classes=num_classes)

    return accuracy, average_precision, average_recall


def compute_validation_loss(model, loss, data_fetcher, device, epoch, writer):
    loss_per_epoch = 0
    model = model.to(device)
    loss = loss.to(device)
    model.eval()
    with torch.no_grad():
        for i, (images, labels, indexes) in enumerate(data_fetcher):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pooled_features, outputs = model(images)
            loss_per_batch = loss(outputs, labels)
            loss_per_epoch += loss_per_batch.item()
            logging.info('Validation Loss' + ' Epoch ' + str(epoch) +
                         ' for minibatch: ' + str(i) + '/' + str(len(data_fetcher))
                         + ' is ' + str(loss_per_batch.item())) \
                if i % 10 == 0 else None
            writer.add_scalar('Loss/MiniBatch_Val', loss_per_batch.item(), epoch*i)
    return loss_per_epoch/len(data_fetcher)


def train_single_epoch(model, loss, optimizer, data_fetcher, device, epoch, writer):
    loss_per_epoch = 0
    model = model.to(device)
    loss = loss.to(device)
    loss.train()
    model.train()
    for i, (images, labels, indexes) in enumerate(data_fetcher):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pooled_features, outputs = model(images)
        loss_per_batch = loss(outputs, labels)
        loss_per_epoch += loss_per_batch.item()

        optimizer.zero_grad()
        loss_per_batch.backward()
        optimizer.step()
        logging.info('Training Loss ' + ' Epoch ' + str(epoch) + ' for minibatch: ' + str(i) + '/' + str(len(data_fetcher)) +
                     ' is ' + str(loss_per_batch.item())) \
            if i % 10 == 0 else None

        writer.add_scalar('Loss/MiniBatch_Train', loss_per_batch.item(), epoch * i)
    return loss_per_epoch/len(data_fetcher)


def main(args):
    # parse configuration file for datasets
    config = config_eval.load_config(config_name=args.config_name)

    experiment_name = os.path.join(config['log']['path'], config['log']['name'])
    if not os.path.isdir(experiment_name):
        os.makedirs(experiment_name)

    # specify summaries directory
    writer = SummaryWriter(experiment_name)

    # TODO - Cleanup this mess to see clearly
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.DEBUG if config['verbose'] else logging.INFO,
        handlers=[
            logging.FileHandler(
                "{0}/{1}/{1}.log".format(
                    config['log']['path'],
                    config['log']['name']
                )
            ),
            logging.StreamHandler()
        ]
    )

    train_transforms = make_transform(**config['transform_parameters'], is_train=True)

    val_tranforms = make_transform(**config['transform_parameters'], is_train=False)

    # define the dataset
    dataset_selected = config['dataset_name']
    train_dataset = Dataset_from_CSV(root=config['dataset'][dataset_selected]['root'],
                                     csv_file=config['dataset'][dataset_selected]['train_csv'],
                                     transform=train_transforms)

    val_dataset = Dataset_from_CSV(root=config['dataset'][dataset_selected]['root'],
                                     csv_file=config['dataset'][dataset_selected]['val_csv'],
                                     transform=val_tranforms)

    # create dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config['sz_batch'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   pin_memory=True,
                                                   drop_last=True, collate_fn=my_collate)

    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=config['sz_batch'],
                                                 shuffle=True,
                                                 num_workers=8,
                                                 pin_memory=True,
                                                 drop_last=True, collate_fn=my_collate)
    num_classes = train_dataset.nb_classes()

    # define model and add to define devices where applicable
    model = model_getter.get_model_for_dml(name=config['model_name'], num_classes=num_classes)

    device = torch.device("cuda:" + str(config['gpu_id']) if torch.cuda.is_available() else "cpu")

    # resume by loading previously best model
    if config['resume']:
        logging.info('Loading Model from the checkpoint:', config['resume_path'])
        model.load_state_dict(torch.load(config['resume_path']))
    weights = train_dataset.get_class_distribution()

    # define loss function
    loss_selected = config['loss_selected']
    criterion = None
    if loss_selected == 'softmax':
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))
    else:
        ValueError('Loss selected is not defined')
        exit(0)

    # create a dictionary to define different learning rates for pretrained backbone and the new layer
    optimizer = torch.optim.Adam([
        {"params": model.pretrained_model.parameters(), "lr": config['optimizer_params']['base_lr']},
        {"params": model.classification_layer.parameters(), "lr": config['optimizer_params']['new_params']}],
        eps=config['optimizer_params']['eps'], amsgrad=True)

    logging.info('Dataset selected:' + config['dataset_name'])
    logging.info('Number of classes in training:' + str(train_dataset.nb_classes()))
    logging.info('Number of Images in training:' + str(len(train_dataset)))
    logging.info('Number of classes in validation:' + str(val_dataset.nb_classes()))
    logging.info('Number of Images in validation:' + str(len(val_dataset)))
    logging.info("Loss selected:" + str(config['loss_selected']))

    validation_loss = 10000
    best_accuracy = 0
    best_epoch = 0

    for epoch in range(1, config['nb_epochs']):
        train_loss = train_single_epoch(model=model, loss=criterion, optimizer=optimizer,
                                        data_fetcher=train_dataloader, device=device,
                                        epoch=epoch, writer=writer)

        training_acc, training_precision, training_recall = evaluate(model=model,
                                                                     test_loader=train_dataloader,
                                                                     device=device, num_classes=num_classes)
        logging.info('Training loss for epoch:' + str(epoch) + ' is: ' + str(train_loss))
        logging.info('Training Accuracy for epoch:' + str(epoch) + ' is: ' + str(training_acc) + '%')

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', training_acc, epoch)
        writer.add_scalar('Precision/train', training_precision)
        writer.add_scalar('Recall/train', training_recall)
        if epoch % config['nb_val_epochs'] == 0:
            val_loss = compute_validation_loss(model=model, loss=criterion, data_fetcher=val_dataloader,
                                               device=device, epoch=epoch, writer=writer)
            accuracy, precision, recall = evaluate(test_loader=val_dataloader, model=model,
                                                   device=device, num_classes=num_classes)

            logging.info('Validation loss at epoch:' + str(epoch) + ' is ' + str(val_loss))
            logging.info('Accuracy for Validation: ' + str(accuracy) + '%')

            writer.add_scalar('Loss/val', train_loss, epoch)
            writer.add_scalar('Accuracy/val', training_acc, epoch)
            writer.add_scalar('Precision/val', precision)
            writer.add_scalar('Recall/val', recall)

            if best_accuracy <= accuracy:
                validation_loss = val_loss
                best_accuracy = accuracy
                best_epoch = epoch
                saving_path = os.path.join(config['log']['path'],
                                           config['log']['name'],
                                           'best_epoch.pth')
                logging.info('Saving Path for best_model: ' + str(saving_path))
                logging.info('Best Epoch at: ' + str(epoch))
                torch.save(model.state_dict(), saving_path)
        logging.info('Best Validation Loss: ' + str(validation_loss))
        logging.info('Best Accuracy: ' + str(best_accuracy))
        logging.info('Best Epoch: ' + str(best_epoch))

    logging.info('---------------------------------------------------------------------------------------------')
    logging.info('---------------------------------------------------------------------------------------------')
    logging.info('Training Completed...')
    logging.info('Best Validation Loss: ' + str(validation_loss))
    logging.info('Best Accuracy: ' + str(best_accuracy))
    logging.info('Best Epoch: ' + str(best_epoch))


if __name__ == '__main__':
    main(parse_args())
