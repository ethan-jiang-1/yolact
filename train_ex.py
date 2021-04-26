#from data import *
from data.config import MEANS, cfg,  set_cfg, set_dataset
from data.coco import COCODetection, detection_collate, enforce_size
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from yolact import Yolact
import os
#import sys
import time
import math, random
#from pathlib import Path
import torch
#from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
#import torch.backends.cudnn as cudnn
#import torch.nn.init as init
import torch.utils.data as data
#import numpy as np
import argparse
import datetime

# Oof
import eval as eval_script

import signal
#import os
#import time

sig_num = None
def receive_signal(signum, stack):
    global sig_num
    sig_num = signum
    print()
    print('#r# Received Signal: {} '.format(signum))
    print()

def get_sig_num():
    return sig_num

signal.signal(signal.SIGUSR1, receive_signal)
#signal.signal(signal.SIGUSR2, receive_signal)


loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']
cur_lr = 1e-5

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Yolact Training Script')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                            ', the model will resume training from the interrupt file.')
    parser.add_argument('--start_iter', default=-1, type=int,
                        help='Resume training at this iter. If this is -1, the iteration will be'\
                            'determined from the file name.')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                        help='Initial learning rate. Leave as None to read this from the config.')
    parser.add_argument('--momentum', default=None, type=float,
                        help='Momentum for SGD. Leave as None to read this from the config.')
    parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                        help='Weight decay for SGD. Leave as None to read this from the config.')
    parser.add_argument('--gamma', default=None, type=float,
                        help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving checkpoint models.')
    parser.add_argument('--log_folder', default='logs/',
                        help='Directory for saving logs.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--save_interval', default=10000, type=int,
                        help='The number of iterations between saving the model.')
    parser.add_argument('--validation_size', default=5000, type=int,
                        help='The number of images to use for validation.')
    parser.add_argument('--validation_epoch', default=2, type=int,
                        help='Output validation information every n iterations. If -1, do no validation.')
    parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                        help='Only keep the latest checkpoint instead of each one.')
    parser.add_argument('--keep_latest_interval', default=100000, type=int,
                        help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--no_log', dest='log', action='store_false',
                        help='Don\'t log per iteration information into log_folder.')
    parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                        help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
    parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                        help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
    parser.add_argument('--batch_alloc', default=None, type=str,
                        help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
    parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                        help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')

    parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
    a_args = parser.parse_args()

    # Update training parameters from the config if necessary
    def replace(name):
        if getattr(a_args, name) is None: setattr(a_args, name, getattr(cfg, name))

    replace('lr')
    replace('decay')
    replace('gamma')
    replace('momentum')

    a_args.saved_epoch = 0
    a_args.saved_iteration = 0
    return a_args

def update_env_and_cfg_by_args(a_args):

    if a_args.config is not None:
        set_cfg(a_args.config)

    if a_args.dataset is not None:
        set_dataset(a_args.dataset)

    if a_args.autoscale and a_args.batch_size != 8:
        factor = a_args.batch_size / 8
        if __name__ == '__main__':
            print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, a_args.batch_size))

        cfg.lr *= factor
        cfg.max_iter //= factor
        cfg.lr_steps = [x // factor for x in cfg.lr_steps]

    # This is managed by set_lr
    set_lr(None, a_args.lr)

    if torch.cuda.device_count() == 0:
        print('No GPUs detected. Exiting...')
        exit(-1)

    if a_args.batch_size // torch.cuda.device_count() < 6:
        if __name__ == '__main__':
            print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
        cfg.freeze_bn = True

    if torch.cuda.is_available():
        if a_args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not a_args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """
    
    def __init__(self, net:Yolact, criterion:MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion
    
    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses

class CustomDataParallel(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, args=None):
        super(CustomDataParallel, self).__init__(module)
        self.args = args

    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], self.args, devices, allocation=self.args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])
        
        return out


def get_default_log_avgs():
    return { k: MovingAverage(100) for k in loss_types }


def prepare_dataset_dataloader(args):
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))
    
    val_dataset = None
    if args.validation_epoch > 0:
        setup_eval(args)
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check    
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    return dataset, val_dataset, data_loader


def prepare_model(args):
    yolact_net = Yolact()
    net = yolact_net
    net.train()

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        init_path = args.save_folder + cfg.backbone.path
        print('Initializing weights...', init_path)
        if os.path.isfile(init_path):
            yolact_net.init_weights(backbone_path=init_path)
        else:
            print("no init weight, use empty")
    return yolact_net


def prepare_loss_optimizer(args, yolact_net):

    optimizer = optim.SGD(yolact_net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    netloss = CustomDataParallel(NetLoss(yolact_net, criterion), args=args)
    if args.cuda:
        netloss = netloss.cuda()
    
    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    return netloss, optimizer

def prepare_log(args):
    log = None
    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
            overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)
    return log



def save_yolact_net(yolact_net, args, iteration, epoch, mode="iteration"):
    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
 
    if mode == "iteration":
        if args.keep_latest:
            latest = SavePath.get_latest(args.save_folder, cfg.name)
    elif mode == "interrupt":
        # Delete previous copy of the interrupted network so we don't spam the weights folder
        SavePath.remove_interrupt(args.save_folder)

    if mode != "interrupt":
        saved_pathname = save_path(epoch, iteration)
    else:
        saved_pathname = save_path(epoch, repr(iteration) + '_interrupt') 
    print('Saving state, iter:', iteration, "saved at", saved_pathname)
    yolact_net.save_weights(saved_pathname)

    if mode == "iteration":
        if args.keep_latest and latest is not None:
            if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                print('Deleting old save...')
                os.remove(latest)
    return saved_pathname

def log_iteration(log, losses, loss, iteration, epoch, elapsed, args):
    precision = 5
    loss_info = {k: round(losses[k].item(), precision) for k in losses}
    loss_info['T'] = round(loss.item(), precision)

    if args.log_gpu:
        log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
        
    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
        lr=round(get_lr(), 10), elapsed=elapsed)

    log.log_gpu_stats = args.log_gpu

def prompt_progress(epoch, iteration, elapsed, time_avg, loss_avgs, losses):
    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]
    
    total = sum([loss_avgs[k].get_avg() for k in losses])
    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
    
    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
            % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)


def update_cfg_lr(iteration, optimizer, loss_avgs, args):
    changed = False
    for change in cfg.delayed_settings:
        if iteration >= change[0]:
            changed = True
            cfg.replace(change[1])

            # Reset the loss averages because things might have changed
            for avg in loss_avgs:
                avg.reset()
    
    # If a config setting was changed, remove it from the list so we don't keep checking
    if changed:
        cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

    # Warm up by linearly interpolating the learning rate from some smaller value
    if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
        set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)


def resume_train_from_saved_model(args, saved_pathname):
    if not os.path.isfile(saved_pathname):
        return False

    print("resume from where break")
    args.resume = saved_pathname
    basename = os.path.basename(saved_pathname)

    names = basename.split(".")[0].split("_")
    epoch = names[len(names)-2]
    iteration = names[len(names)-1]

    try:
        epoch = int(epoch)
        iteration = int(iteration)

        print("saved_epoch", epoch)
        print("saved_iteration", iteration)
        args.saved_epoch = epoch
        args.saved_iteration = iteration
    except:
        pass


def train(args, dataset, val_dataset, data_loader, yolact_net, netloss, optimizer, log):
    # loss counters
    #loc_loss = 0
    #conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()
    
    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    time_avg = MovingAverage()

    #global loss_types # Forms the print order
    loss_avgs  = get_default_log_avgs()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)

    start_epoch = 0
    if args.resume is not None:
        start_epoch = args.saved_epoch

    print('Begin training!')
    print("start_epoch", start_epoch)
    print("num_epochs", num_epochs)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Resume from start_iter
        if (epoch+1)*epoch_size < iteration:
            continue
        
        for datum in data_loader:
            # Stop if we've reached an epoch if we're resuming from start_iter
            if iteration == (epoch+1)*epoch_size:
                break

            # Stop at the configured number of iterations even if mid-epoch
            if iteration == cfg.max_iter:
                break

            update_cfg_lr(iteration, optimizer, loss_avgs, args)

            # Warm up by linearly interpolating the learning rate from some smaller value
            if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

            # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
            while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                step_index += 1
                set_lr(optimizer, args.lr * (args.gamma ** step_index))
            
            # Zero the grad to get ready to compute gradients
            optimizer.zero_grad()

            # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
            losses = netloss(datum)
            
            losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
            loss = sum([losses[k] for k in losses])
            
            # no_inf_mean removes some components from the loss, so make sure to backward through all of it
            # all_loss = sum([v.mean() for v in losses.values()])

            # Backprop
            loss.backward() # Do this to free up vram even if loss is not finite
            if torch.isfinite(loss).item():
                optimizer.step()
            
            # Add the loss to the moving average for bookkeeping
            for k in losses:
                loss_avgs[k].add(losses[k].item())

            cur_time  = time.time()
            elapsed   = cur_time - last_time
            last_time = cur_time

            # Exclude graph setup from the timing information
            if iteration != args.start_iter:
                time_avg.add(elapsed)

            if iteration % 10 == 0:
                prompt_progress(epoch, iteration, elapsed, time_avg, loss_avgs, losses)

            if args.log:
                log_iteration(log, losses, loss, iteration, epoch, elapsed, args)
            
            iteration += 1

            if iteration % args.save_interval == 0 and iteration != args.start_iter:
                save_yolact_net(yolact_net, args, iteration, epoch)
        
        # This is done per epoch
        if args.validation_epoch > 0:
            if epoch % args.validation_epoch == 0 and epoch > 0:
                compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)

        if sig_num is not None:
            print("#r# break traning loop due to sig_num", sig_num)
            break
    
    # Compute validation mAP after training is finished
    compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)

    saved_pathname = None
    if sig_num is not None:
        saved_pathname = save_yolact_net(yolact_net, args, iteration, epoch, mode="sig_num")
    else:
        saved_pathname = save_yolact_net(yolact_net, args, iteration, epoch, mode="finial")

    return saved_pathname 


def set_lr(optimizer, new_lr):
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr

def get_lr():
    return cur_lr

def gradinator(x):
    x.requires_grad = False
    return x

def prepare_data(datum, args, devices:list=None, allocation:list=None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less
        
        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx]  = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images)-1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)
        
        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds

def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()

def compute_validation_loss(net, data_loader, criterion, args):
    global loss_types

    with torch.no_grad():
        losses = {}
        
        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum, args)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())
            
            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break
        
        for k in losses:
            losses[k] /= iterations
            
        
        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)


def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None):
    with torch.no_grad():
        yolact_net.eval()
        
        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()

def setup_eval(args):
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])

def reset_kill_signal():
    global sig_num
    sig_num = None
    print("reset sig_num")

def prompt_kill_signal():
    pid = os.getpid()
    print()
    print('#m# start train_ex PID: {}'.format(pid))
    print("run")
    print("kill -n 10 {}".format(pid)) 
    print("to fire signal 10 to this process")
    print()


if __name__ == '__main__':
    prompt_kill_signal()

    args = parse_args()
    update_env_and_cfg_by_args(args)

    dataset, val_dataset, data_loader= prepare_dataset_dataloader(args)
    yolact_net = prepare_model(args)
    netloss, optimizer = prepare_loss_optimizer(args, yolact_net)
    log = prepare_log(args)

    train(args, dataset, val_dataset, data_loader, yolact_net, netloss, optimizer, log)

