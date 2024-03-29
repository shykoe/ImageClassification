from __future__ import print_function, division, absolute_import
import argparse
import os
import shutil
import time
import pdb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys
import logging
logger = logging.getLogger('train')
import models
import utils
model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Modelzoo Training')
parser.add_argument('--learning-rate-decay-start', type=int, default=-1, 
help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')

parser.add_argument('--learning-rate-decay-every', type=int, default=3, 
help='every how many iterations thereafter to drop LR?(in epoch)')

parser.add_argument('--learning-rate-decay-rate', type=float, default=0.8, 
help='every how many iterations thereafter to drop LR?(in epoch)')

parser.add_argument('--num-classes', type=int, default=1000, 
help='number of classes in dataset, default 1000')

parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'sgd'],
                    help='optim to choose (adam) ')
parser.add_argument('--train-data', metavar='DIR', default="/home/shykoe/finetune/",
                    help='path to train dataset')
parser.add_argument('--val-data', metavar='DIR', default="/home/shykoe/finetune/",
                    help='path to val dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='se_resnext50_32x4d',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: fbresnet152)')
parser.add_argument('--save-path', default='/home/shykoe/finetune', type=str,
                    help='save path')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0004, type=float,
                    metavar='LR', help='initial learning rate ,default 4e-4')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=False,
                    action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', default='no', help='use pre-trained model')
parser.add_argument('--do-not-preserve-aspect-ratio',
                    dest='preserve_aspect_ratio',
                    help='do not preserve the aspect ratio when resizing an image',
                    action='store_false')
parser.set_defaults(preserve_aspect_ratio=True)
best_prec1 = 0
def get_params():
    global args, best_prec1
    args, _ = parser.parse_known_args()
    return args


def main(args):
    global best_prec1
    is_cuda = torch.cuda.is_available()
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.pretrained.lower() not in ['false', 'none', 'not', 'no', '0']:
        print("=> using pre-trained parameters '{}'".format(args.pretrained))
        model = models.__dict__[args.arch](num_classes=args.num_classes,
                                                     pretrained='imagenet')
    else:
        model = models.__dict__[args.arch](num_classes=args.num_classes, pretrained=None)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    scale = 0.875
    print('Images transformed from size {} to {}'.format(
        int(round(max(model.input_size) / scale)),
        model.input_size))
    train_tf = transforms.Compose([
        transforms.RandomSizedCrop(max(model.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])
    val_tf = transforms.Compose([
        transforms.Resize((max(model.input_size),max(model.input_size))),
        transforms.ToTensor()
        ]
    )



    train_dataset = datasets.ImageFolder(args.train_data,train_tf)
    val_dataset = datasets.ImageFolder(args.val_data,val_tf)


    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=args.batch_size,
                    shuffle=True, 
                    num_workers=args.workers,
                    pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                    batch_size=args.batch_size,
                    shuffle=True, 
                    num_workers=args.workers,
                    pin_memory=True)
    # define loss function (criterion) and optimizer
    if is_cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                #momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
       
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                #momentum=args.momentum,
                                weight_decay=args.weight_decay)
    else:
        raise
    if is_cuda:
        model = torch.nn.DataParallel(model).cuda() 
        #model = model.cuda() 


    for epoch in range(args.start_epoch, args.epochs):
        if args.learning_rate_decay_start != -1 and epoch > args.learning_rate_decay_start:
            adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        
        prec1,prec5 = validate(val_loader, model, criterion)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            target = target.cuda()
            input = input.cuda()
        #target = target
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        try:
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
        except Exception as e :
            raise    
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                input = input.cuda()

            #target = target
            #input = input
            # compute output
            output = model(input)
            loss = criterion(output, target)
            #import pdb; pdb.set_trace()
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(args.save_path, filename))
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.learning_rate_decay_rate ** (epoch // args.learning_rate_decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    try:
        print (time.strftime('%Y-%m-%d %A %X %Z',time.localtime(time.time())) )
        starttime = time.time()
        main(get_params())
        print('{}'.format(time.time()-starttime))
    except Exception as exception:
        logger.exception(exception)
        raise
        
