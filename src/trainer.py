import torch
import time
import torch.nn as nn
import numpy as np
from sacred import Ingredient
from src.utils import warp_tqdm, get_metric, AverageMeter
from src.datasets.ingredient import get_dataloader

trainer_ingredient = Ingredient('trainer')
@trainer_ingredient.config
def config():
    print_freq = 10
    meta_val_way = 5
    meta_val_shot = 1
    meta_val_metric = 'cosine'  # ('euclidean', 'cosine', 'l1', l2')
    meta_val_iter = 500
    meta_val_query = 15
    alpha = - 1.0
    label_smoothing = 0.


class Trainer:
    @trainer_ingredient.capture
    def __init__(self, device, meta_val_iter, meta_val_way, meta_val_shot,
                 meta_val_query, ex):

        self.train_loader = get_dataloader(split='train', aug=True, shuffle=True)
        sample_info = [meta_val_iter, meta_val_way, meta_val_shot, meta_val_query]
        self.val_loader = get_dataloader(split='val', aug=False, sample=sample_info)
        self.device = device
        self.num_classes = ex.current_run.config['model']['num_classes']

    def cross_entropy(self, logits, one_hot_targets, reduction='batchmean'):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        return - (one_hot_targets * logsoftmax).sum(1).mean()

    @trainer_ingredient.capture
    def do_epoch(self, epoch, scheduler, print_freq, disable_tqdm, callback, model,
                 alpha, optimizer):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to train mode
        model.train()
        steps_per_epoch = len(self.train_loader)
        end = time.time()
        tqdm_train_loader = warp_tqdm(self.train_loader, disable_tqdm)
        for i, (input, target, _) in enumerate(tqdm_train_loader):

            input, target = input.to(self.device), target.to(self.device, non_blocking=True)

            smoothed_targets = self.smooth_one_hot(target)
            assert (smoothed_targets.argmax(1) == target).float().mean() == 1.0
            # Forward pass
            if alpha > 0:  # Mixup augmentation
                # generate mixed sample and targets
                lam = np.random.beta(alpha, alpha)
                rand_index = torch.randperm(input.size()[0]).cuda()
                target_a = smoothed_targets
                target_b = smoothed_targets[rand_index]
                mixed_input = lam * input + (1 - lam) * input[rand_index]

                output = model(mixed_input)
                loss = self.cross_entropy(output, target_a) * lam + self.cross_entropy(output, target_b) * (1. - lam)
            else:
                output = model(input)
                loss = self.cross_entropy(output, smoothed_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = (output.argmax(1) == target).float().mean()
            top1.update(prec1.item(), input.size(0))
            if not disable_tqdm:
                tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

            # Measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(self.train_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
                if callback is not None:
                    callback.scalar('train_loss', i / steps_per_epoch + epoch, losses.avg, title='Train loss')
                    callback.scalar('@1', i / steps_per_epoch + epoch, top1.avg, title='Train Accuracy')
        scheduler.step()
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        if callback is not None:
            callback.scalar('lr', epoch, current_lr, title='Learning rate')

    @trainer_ingredient.capture
    def smooth_one_hot(self, targets, label_smoothing):
        assert 0 <= label_smoothing < 1
        with torch.no_grad():
            new_targets = torch.empty(size=(targets.size(0), self.num_classes), device=self.device)
            new_targets.fill_(label_smoothing / (self.num_classes-1))
            new_targets.scatter_(1, targets.unsqueeze(1), 1. - label_smoothing)
        return new_targets

    @trainer_ingredient.capture
    def meta_val(self, model, meta_val_way, meta_val_shot, disable_tqdm, callback, epoch):
        top1 = AverageMeter()
        model.eval()

        with torch.no_grad():
            tqdm_test_loader = warp_tqdm(self.val_loader, disable_tqdm)
            for i, (inputs, target, _) in enumerate(tqdm_test_loader):
                inputs, target = inputs.to(self.device), target.to(self.device, non_blocking=True)
                output = model(inputs, feature=True)[0].cuda(0)
                train_out = output[:meta_val_way * meta_val_shot]
                train_label = target[:meta_val_way * meta_val_shot]
                test_out = output[meta_val_way * meta_val_shot:]
                test_label = target[meta_val_way * meta_val_shot:]
                train_out = train_out.reshape(meta_val_way, meta_val_shot, -1).mean(1)
                train_label = train_label[::meta_val_shot]
                prediction = self.metric_prediction(train_out, test_out, train_label)
                acc = (prediction == test_label).float().mean()
                top1.update(acc.item())
                if not disable_tqdm:
                    tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg * 100))

        if callback is not None:
            callback.scalar('val_acc', epoch + 1, top1.avg, title='Val acc')
        return top1.avg

    @trainer_ingredient.capture
    def metric_prediction(self, support, query, train_label, meta_val_metric):
        support = support.view(support.shape[0], -1)
        query = query.view(query.shape[0], -1)
        distance = get_metric(meta_val_metric)(support, query)
        predict = torch.argmin(distance, dim=1)
        predict = torch.take(train_label, predict)
        return predict