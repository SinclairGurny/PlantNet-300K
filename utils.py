import torch
import torch.nn as nn
import random
import timm
import numpy as np
import os
from collections import Counter

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from torchvision.transforms import CenterCrop


def set_seed(args, use_gpu, print_out=True):
    if print_out:
        print('Seed:\t {}'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed(args.seed)


def update_correct_per_class(batch_output, batch_y, d):
    predicted_class = torch.argmax(batch_output, dim=-1)
    for true_label, predicted_label in zip(batch_y, predicted_class):
        if true_label == predicted_label:
            d[true_label.item()] += 1
        else:
            d[true_label.item()] += 0


def update_correct_per_class_topk(batch_output, batch_y, d, k):
    topk_labels_pred = torch.argsort(batch_output, axis=-1, descending=True)[:, :k]
    for true_label, predicted_labels in zip(batch_y, topk_labels_pred):
        d[true_label.item()] += torch.sum(true_label == predicted_labels).item()


def update_correct_per_class_avgk(val_probas, val_labels, d, lmbda):
    ground_truth_probas = torch.gather(val_probas, dim=1, index=val_labels.unsqueeze(-1))
    for true_label, predicted_label in zip(val_labels, ground_truth_probas):
        d[true_label.item()] += (predicted_label >= lmbda).item()


def count_correct_topk(scores, labels, k):
    """Given a tensor of scores of size (n_batch, n_classes) and a tensor of
    labels of size n_batch, computes the number of correctly predicted exemples
    in the batch (in the top_k accuracy sense).
    """
    top_k_scores = torch.argsort(scores, axis=-1, descending=True)[:, :k]
    labels = labels.view(len(labels), 1)
    return torch.eq(labels, top_k_scores).sum()


def count_correct_avgk(probas, labels, lmbda):
    """Given a tensor of scores of size (n_batch, n_classes) and a tensor of
    labels of size n_batch, computes the number of correctly predicted exemples
    in the batch (in the top_k accuracy sense).
    """
    gt_probas = torch.gather(probas, dim=1, index=labels.unsqueeze(-1))
    res = torch.sum((gt_probas) >= lmbda)
    return res


def load_model(model, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError

    device = 'cuda:0' if use_gpu else 'cpu'
    d = torch.load(filename, map_location=device)
    model.load_state_dict(d['model'])
    return d['epoch']


def load_optimizer(optimizer, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError

    device = 'cuda:0' if use_gpu else 'cpu'
    d = torch.load(filename, map_location=device)
    optimizer.load_state_dict(d['optimizer'])


def save(model, optimizer, epoch, location):
    dir = os.path.dirname(location)
    if not os.path.exists(dir):
        os.makedirs(dir)

    d = {'epoch': epoch,
         'model': model.state_dict(),
         'optimizer': optimizer.state_dict()}
    torch.save(d, location)


def decay_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    print('Switching lr to {}'.format(optimizer.param_groups[0]['lr']))
    return optimizer


def update_optimizer(optimizer, lr_schedule, epoch):
    if epoch in lr_schedule:
        optimizer = decay_lr(optimizer)
    return optimizer


def get_model(args, n_classes):
    timm_models = timm.list_models()

    if args.model in timm_models:
        model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=n_classes)
    else:
        raise NotImplementedError
        
    if args.model_file is not None and len(args.model_file) > 0:
        load_model(model, args.model_file, args.use_gpu)
    return model


class Plantnet(ImageFolder):
    def __init__(self, root, split, **kwargs):
        self.root = root
        self.split = split
        super().__init__(self.split_folder, **kwargs)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)


def get_data(root, image_size, crop_size, batch_size, num_workers, pretrained):

    if pretrained:
        transform_train = transforms.Compose([transforms.Resize(size=image_size), transforms.RandomCrop(size=crop_size),
                                              transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                          std=[0.229, 0.224, 0.225])])
        transform_test = transforms.Compose([transforms.Resize(size=image_size), transforms.CenterCrop(size=crop_size),
                                             transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])])
    else:
        transform_train = transforms.Compose([transforms.Resize(size=image_size), transforms.RandomCrop(size=crop_size),
                                              transforms.ToTensor(), transforms.Normalize(mean=[0.4425, 0.4695, 0.3266],
                                                                                          std=[0.2353, 0.2219, 0.2325])])
        transform_test = transforms.Compose([transforms.Resize(size=image_size), transforms.CenterCrop(size=crop_size),
                                             transforms.ToTensor(), transforms.Normalize(mean=[0.4425, 0.4695, 0.3266],
                                                                                         std=[0.2353, 0.2219, 0.2325])])

    trainset = Plantnet(root, 'train', transform=transform_train)
    train_class_to_num_instances = Counter(trainset.targets)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    valset = Plantnet(root, 'val', transform=transform_test)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)

    testset = Plantnet(root, 'test', transform=transform_test)
    test_class_to_num_instances = Counter(testset.targets)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    val_class_to_num_instances = Counter(valset.targets)
    n_classes = len(trainset.classes)

    dataset_attributes = {'n_train': len(trainset), 'n_val': len(valset), 'n_test': len(testset), 'n_classes': n_classes,
                          'class2num_instances': {'train': train_class_to_num_instances,
                                                  'val': val_class_to_num_instances,
                                                  'test': test_class_to_num_instances},
                          'class_to_idx': trainset.class_to_idx}

    return trainloader, valloader, testloader, dataset_attributes
