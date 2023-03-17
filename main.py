"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
import os
import glob

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as tf

import models
import datasets
from pytorch_msssim import ssim
from utils import GradientLoss


def apply_model(net, lres, guide, ensemble=False):
    blur = tf.GaussianBlur(kernel_size=(3,3))
    out = net(lres, guide)

    if ensemble:
        lres_90 = torch.rot90(lres, k=1, dims=[2,3])
        guide_90 = torch.rot90(guide, k=1, dims=[2,3]) 
        out_90 = net(lres_90, guide_90)
        out += torch.rot90(out_90, k=3, dims=[2,3])

        lres_180 = torch.rot90(lres, k=2, dims=[2,3])
        guide_180 = torch.rot90(guide, k=2, dims=[2,3]) 
        out_180 = net(lres_180, guide_180)
        out += torch.rot90(out_180, k=2, dims=[2,3])
        
        lres_270 = torch.rot90(lres, k=3, dims=[2,3])
        guide_270 = torch.rot90(guide, k=3, dims=[2,3]) 
        out_270 = net(lres_270, guide_270)
        out += torch.rot90(out_270, k=1, dims=[2,3])

        # v-flip
        lres = torch.flipud(lres)
        guide = torch.flipud(guide)
        out_vflip = net(lres, guide)
        out += torch.flipud(out_vflip)

        # augs-set2 on flipped image
        lres_90 = torch.rot90(lres, k=1, dims=[2, 3])
        guide_90 = torch.rot90(guide, k=1, dims=[2, 3])
        out_90 = net(lres_90, guide_90)
        out += torch.flipud(torch.rot90(out_90, k=3, dims=[2, 3]))

        lres_180 = torch.rot90(lres, k=2, dims=[2, 3])
        guide_180 = torch.rot90(guide, k=2, dims=[2, 3])
        out_180 = net(lres_180, guide_180)
        out += torch.flipud(torch.rot90(out_180, k=2, dims=[2, 3]))

        lres_270 = torch.rot90(lres, k=3, dims=[2, 3])
        guide_270 = torch.rot90(guide, k=3, dims=[2, 3])
        out_270 = net(lres_270, guide_270)
        out += torch.flipud(torch.rot90(out_270, k=1, dims=[2, 3]))

        # extra augs:
        # blur
        guide_blur = blur(guide)
        out += net(lres, guide_blur)
        
        return out/9.0
    return out

def train(model, train_loader, optimizer, device, epoch, lr, loss_type, perf_measures, args, writer):
    model.train()
    log = []
    running_loss=0
    
    # Can be taken outside to prevent re-initialization every epoch
    # Losses are functional (usually) and dont have any params so can be on CPU.
    grad_loss = GradientLoss()

    for batch_idx, sample in enumerate(train_loader):
        lres, guide, target, filename = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[3]

        optimizer.zero_grad()
        output = apply_model(model, lres, guide)

        # TODO: add SSIM loss and loss that improves PSNR
        if loss_type == 'l2':
            loss = F.mse_loss(output, target)
        elif loss_type == 'l1':
            loss = F.l1_loss(output, target)
        elif loss_type == 'ssim':
            loss = 1-ssim(output, target, data_range=1, size_average=True, nonnegative_ssim=True)
        elif loss_type == 'grad':
            loss = 1-ssim(output, target, data_range=1, size_average=True, nonnegative_ssim=True)
            loss += 2.0*grad_loss(output, target)
        elif loss_type == 'l1_ssim':
            loss = F.l1_loss(output, target)
            loss += 1-ssim(output, target, data_range=1, size_average=True, nonnegative_ssim=True)
        elif loss_type == 'epe':
            loss = models.th_epe(output, target)
        elif loss_type == 'rmse':
            loss = models.th_rmse(output, target)
        else:
            raise ValueError('Loss type ({}) not supported.'.format(args.loss))

        loss.backward()
        optimizer.step()

        batch_cnt = batch_idx + 1
        sample_cnt = batch_idx * args.batch_size + len(lres)
        running_loss += loss.item()

        progress = sample_cnt / len(train_loader.dataset)

        if batch_cnt == len(train_loader) or batch_cnt % args.log_interval == 0:
            '''
            writer.add_image('input', torchvision.utils.make_grid(lres), epoch*len(train_loader) + batch_idx)
            writer.add_image('guide', torchvision.utils.make_grid(guide), epoch*len(train_loader) + batch_idx)
            writer.add_image('target', torchvision.utils.make_grid(target), epoch*len(train_loader) + batch_idx)
            writer.add_image('output', torchvision.utils.make_grid(output), epoch*len(train_loader) + batch_idx)
            '''
            writer.add_scalar('train_loss', running_loss/args.log_interval, epoch*len(train_loader) + batch_idx)
            running_loss = 0.0

            log_row = [progress + epoch - 1, lr, loss.item()]
            for m in perf_measures:
                if m == 'epe':
                    log_row.append(models.th_epe(output, target).item())
                elif m == 'rmse':
                    log_row.append(models.th_rmse(output, target).item())
            log.append(log_row)

        if batch_cnt == len(train_loader) or batch_cnt % args.print_interval == 0:
            print('Train Epoch {} [{}/{} ({:3.0f}%)]\tLR: {:g}\tLoss: {:.6f}\t'.format(
                epoch, sample_cnt, len(train_loader.dataset), 100. * progress, lr, loss.item()))
    return log


def test(model, test_loader, device, epoch, lr, loss_type, perf_measures, args, writer):
    model.eval()
    loss_accum = 0
    perf_measures_accum = [0.0] * len(perf_measures)

    if args.dump_outputs:
        dump_foldername = os.path.join(args.dump_path, 'output')
        os.makedirs(dump_foldername, exist_ok=True)
   
    # Testing only on l2-loss
    loss_type = 'l2' 
    with torch.no_grad():
        for sample in test_loader:
            lres, guide, target, filename = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[3]

            output = apply_model(model, lres, guide, args.ensemble)
            # TODO: Check the effect of clamping before calculating performance metrics.
            # Clamping helps with better outputs and better PSNR, may not be great for SSIM.
            output = torch.clamp(output, min=0.0, max=1.0)

            if loss_type == 'l2':
                loss = F.mse_loss(output, target)
            elif loss_type == 'epe':
                loss = models.th_epe(output, target)
            elif loss_type == 'rmse':
                loss = models.th_rmse(output, target)
            else:
                raise ValueError('Loss type ({}) not supported.'.format(args.loss))

            loss_accum += loss.item() * len(output)

            # TODO: Add PSNR and SSIM metrics.
            for i, m in enumerate(perf_measures):
                if m == 'epe':
                    perf_measures_accum[i] += models.th_epe(output, target).item() * len(output)
                elif m == 'rmse':
                    perf_measures_accum[i] += models.th_rmse(output, target).item() * len(output)
                elif m == 'psnr':
                    perf_measures_accum[i] += models.th_psnr(output, target).item() * len(output)
                elif m == 'ssim':
                    perf_measures_accum[i] += models.th_ssim(output, target).item() * len(output)

            # Check the min/max values from output tensors
            # print("min max:", torch.min(output), torch.max(output))
            if args.dump_outputs:
                dump_filename = os.path.join(dump_foldername, filename[0])
                dump_img = Image.fromarray((255.0*output[0,0,:,:].cpu().detach().numpy()).astype(np.uint8))
                dump_img.save(dump_filename)
                
                ''' 
                dump_filename = os.path.join(dump_foldername, 'input_'+filename[0])
                dump_img = Image.fromarray((255.0*lres[0,0,:,:].cpu().detach().numpy()).astype(np.uint8))
                dump_img.save(dump_filename)

                dump_filename = os.path.join(dump_foldername, 'guide_'+filename[0])
                dump_img = Image.fromarray((255.0*guide[0,:,:,:].permute(1,2,0).cpu().detach().numpy()).astype(np.uint8))
                dump_img.save(dump_filename)
                
                dump_filename = os.path.join(dump_foldername, 'label_'+filename[0])
                dump_img = Image.fromarray((255.0*target[0,0,:,:].cpu().detach().numpy()).astype(np.uint8))
                dump_img.save(dump_filename)
                ''' 

    test_loss = loss_accum / len(test_loader.dataset)
    log = [float(epoch), lr, test_loss]
    msg = 'Average loss: {:.6f}\n'.format(test_loss)
    for m, mv in zip(perf_measures, perf_measures_accum):
        avg = mv / len(test_loader.dataset)
        msg += '{}: {:.6f}\n'.format(m, avg)
        log.append(avg)
    print('\nTesting (#epochs={})'.format(epoch))
    print(msg)

    writer.add_scalar('val_loss', test_loss, epoch)
    writer.add_scalar('val_psnr', perf_measures_accum[0]/len(test_loader.dataset), epoch)
    writer.add_scalar('val_ssim', perf_measures_accum[1]/len(test_loader.dataset), epoch)
    # return loss to decide intermediate checkpoint saving.
    # TODO: Decide based on validation metrics - PSNR or SSIM or either.
    return [log], test_loss, perf_measures_accum[0]/len(test_loader.dataset), perf_measures_accum[1]/len(test_loader.dataset)


def prepare_log(log_path, header, last_epoch=0):
    # keep all existing log lines up to epoch==last_epoch (included)
    try:
        log = np.genfromtxt(log_path, delimiter=',', skip_header=1, usecols=(0,))
    except:
        log = []
    if len(log) > 0:
        idxs = np.where(log <= last_epoch)[0]
        if len(idxs) > 0:
            lines_to_keep = max(idxs) + 2
            with open(log_path) as f:
                lines = f.readlines()
            with open(log_path, 'w') as f:

                f.writelines(lines[:lines_to_keep])
            return

    with open(log_path, 'w') as f:
        f.write(header + '\n')


def main():
    parser = argparse.ArgumentParser(description='Joint upsampling',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='train', metavar='MODE',
                        help='training or testing mode')
    parser.add_argument('--factor', type=int, default=8, metavar='R',
                        help='upsampling factor')
    parser.add_argument('--data-root', type=str, default='data', metavar='D',
                        help='place to find (or download) data')
    parser.add_argument('--train-filenames', type=str, default='train_list.txt', metavar='TL',
                        help='filenames list of train set')
    parser.add_argument('--val-filenames', type=str, default='val_list.txt', metavar='VL',
                        help='filenames list of val set')
    parser.add_argument('--test-filenames', type=str, default='test_list.txt', metavar='TESTL',
                        help='filenames list of test set')
    parser.add_argument('--exp-root', type=str, default='exp', metavar='E',
                        help='place to save results')
    parser.add_argument('--dump-path', type=str, default='dump', metavar='DP',
                        help='place to save outputs')
    parser.add_argument('--load-weights', type=str, default='', metavar='L',
                        help='file with pre-trained weights')
    parser.add_argument('--model', type=str, default='PacJointUpsample', metavar='M',
                        help='network model type')
    parser.add_argument('--interpolation', type=str, default='bilinear', metavar='INTER',
                        help='interpolation mode for the residual upsampling')
    parser.add_argument('--loss', type=str, default='l2', metavar='L',
                        help='choose a loss function type')
    parser.add_argument('--measures', nargs='+', default=None, metavar='M',
                        help='performance measures to be reported during training and testing')
    parser.add_argument('--num-data-worker', type=int, default=4, metavar='W',
                        help='number of subprocesses for data loading')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--train-crop', type=int, default=40, metavar='CROP',
                        help='input crop size in training')
    parser.add_argument('--epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                        help='pick which optimizer to use')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-steps', nargs='+', default=None, metavar='S',
                        help='decrease lr by 10 at these epochs')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='ignore existing log files and snapshots')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='WD',
                        help='Adam/SGD weight decay')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed')
    parser.add_argument('--print-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before displaying training status')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test-interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before a testing')
    parser.add_argument('--snapshot-interval', type=int, default=1, metavar='N',
                        help='snapshot intermediate models')
    parser.add_argument('--dump-outputs', default=False, action='store_true',
                        help='dump output SR thermal images from evaluation/test')
    parser.add_argument('--num-layers', type=int, default=3, metavar='NL',
                        help='number of residual blocks in feature extraction.')
    parser.add_argument('--num-channels', type=int, default=64, metavar='NC',
                        help='number of channels in each layer.')
    parser.add_argument('--ensemble', default=False, action='store_true',
                        help='test time augmentations')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    assert(torch.cuda.is_available())
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    dl_kwargs = {'num_workers': args.num_data_worker, 'pin_memory': True} if use_cuda else {}

    # find existing snapshots
    os.makedirs(args.exp_root, exist_ok=True)
    snapshots_found = sorted([int(s.split('_')[-1].rstrip('.pth'))
                              for s in glob.glob(os.path.join(args.exp_root, 'weights_epoch_*.pth'))])
    load_weights = args.load_weights
    if snapshots_found and not args.overwrite:
        last_epoch = max(snapshots_found) if args.epochs > max(snapshots_found) else args.epochs
        assert last_epoch in snapshots_found
        assert not load_weights
        load_weights = os.path.join(args.exp_root, 'weights_epoch_{}.pth'.format(last_epoch))
    else:
        last_epoch = 0
    test_only = (args.epochs <= last_epoch)

    # dataset
    ch, guide_ch = 1, 3
    perf_measures = ('psnr', 'ssim') if not args.measures else args.measures       # Validation metrics.

    # TODO: Add random rotation in train
    train_transform = datasets.AssembleJointUpsamplingInputs(flip=True, rotate=True, crop=
                                                                 (None if args.train_crop <= 0 else args.train_crop))
    test_transform = datasets.AssembleJointUpsamplingInputs(flip=False, crop=None, rotate=False)
    if args.epochs > 0:
        train_dset = datasets.PBVS2(args.data_root, split='train', file_names_path=args.train_filenames,
                                    transform=train_transform)
    else:
        train_dset = None
    if args.mode == 'test':
        test_dset = datasets.PBVS2(args.data_root, split='test', file_names_path=args.test_filenames,
                                   transform=test_transform)
    else:
        test_dset = datasets.PBVS2(args.data_root, split='val', file_names_path=args.val_filenames,
                                   transform=test_transform)

    # data loader
    if test_only:
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.test_batch_size, shuffle=True, **dl_kwargs)

    # model
    model = models.__dict__[args.model](args, channels=ch, guide_channels=guide_ch, factor=args.factor)
    if load_weights:
        model.load_state_dict(torch.load(load_weights))
        print('\nModel weights initialized from: {}'.format(load_weights))
    model = model.to(device)

    # optimizer, scheduler, and logs
    if not test_only:
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError('Optimizer type ({}) is not supported.'.format(args.optimizer))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   [] if not args.lr_steps else [int(v) for v in args.lr_steps],
                                                   gamma=0.5,
                                                   last_epoch=-1)

        for s in range(last_epoch):
            scheduler.step()  # TODO： a temporary workaround -- ideally should recover optimizer from checkpoint instead

        # log files
        fmtstr = '{:.6f},{:g},{:.6f},{:.6f}'
        csv_header = 'epoch,lr,loss'
        for m in perf_measures:
            csv_header += (',' + m)

        train_log_path = os.path.join(os.path.join(args.exp_root, 'train.log'))
        test_log_path = os.path.join(os.path.join(args.exp_root, 'test.log'))
        prepare_log(train_log_path, csv_header, last_epoch)
        prepare_log(test_log_path, csv_header, last_epoch)

    # main computation
    writer = SummaryWriter(args.exp_root)
    init_lr = 0 if test_only else scheduler.get_last_lr()[0]
    log_test, _, max_psnr, max_ssim = test(model, test_loader, device, last_epoch, init_lr, args.loss, perf_measures, args, writer)
    if last_epoch == 0 and not test_only:
        with open(test_log_path, 'a') as f:
            f.writelines([','.join([('' if arg == -1 else fmt.format(arg)) for fmt, arg in zip(fmtstr.split(','), l)])
                          + '\n' for l in log_test])

    # Validation metric/loss for checkpoint saving
    for epoch in range(last_epoch + 1, args.epochs + 1):

        lr = scheduler.get_last_lr()[0]
        log_train = train(model, train_loader, optimizer, device, epoch, lr, args.loss, perf_measures, args, writer)

        with open(train_log_path, 'a') as f:
            f.writelines([','.join([('' if arg == -1 else fmt.format(arg)) for fmt, arg in zip(fmtstr.split(','), l)])
                          + '\n' for l in log_train])

        if epoch % args.test_interval == 0:
            log_test, epoch_val_loss, epoch_psnr, epoch_ssim = test(model, test_loader, device, epoch, lr, args.loss, perf_measures, args, writer)
            with open(test_log_path, 'a') as f:
                f.writelines(
                    [','.join([('' if arg == -1 else fmt.format(arg)) for fmt, arg in zip(fmtstr.split(','), l)]) + '\n'
                     for l in log_test])
        scheduler.step()

        if epoch_psnr > max_psnr or epoch_ssim > max_ssim:
            max_psnr = max(epoch_psnr, max_psnr)
            max_ssim = max(epoch_ssim, max_ssim)

            save_weights = os.path.join(args.exp_root, 'weights_epoch_{}.pth'.format(epoch))
            torch.save(model.to('cpu').state_dict(), save_weights)
            print('Snapshot saved to: {}\n'.format(save_weights))
            model = model.to(device)
    writer.close()


if __name__ == '__main__':
    main()
