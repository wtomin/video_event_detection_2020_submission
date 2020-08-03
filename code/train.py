import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from data.dataloader import DataLoader
from torch.autograd import Variable as Variable
from models.MLP_RNN import MLP_RNN
from sklearn.metrics import f1_score
from tqdm import tqdm
parser = argparse.ArgumentParser(description="PyTorch implementation of video event detection")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--lmdb', type=str, default ='../user_data/Train/i3d_features.lmdb')
parser.add_argument('--hidden_units', default=[1024, 256, 256], type=int, nargs="+",
                    help='hidden units set up')
parser.add_argument('--val_ratio', type=float, default = 0.2, help="the validation set ratio")
# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--early_stop', type=int, default=3) # if validation loss didn't improve over 5 epochs, stop
parser.add_argument('-b', '--batch_size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--lr_steps', default=[ 5, 10], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--optim', type=str, choices = ['SGD', 'Adam'], default = 'Adam')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: 20)')
# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=250, type=int,
                    metavar='N', help='print frequency (default: 50) iteration')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 50) epochs')
# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='model')
parser.add_argument('--root_output',type=str, default='output')
parser.add_argument('--root_tensorboard', type=str, default='runs')
args = parser.parse_args()
def averaged_f1_score(input, target):
    N, label_size = input.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(input[:, i], target[:, i])
        f1s.append(f1)
    return np.mean(f1s)

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

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output, args.root_tensorboard]
    folders_util = ["%s/"%(args.save_root) +folder for folder in folders_util]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder)
def train(dataloader, model, criterion, optimizer, epoch, log): 
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    optimizer.zero_grad()
    model.train()
    targets = []
    for i, data_batch in enumerate(dataloader):
        data_time.update(time.time() - end)
        img , label  = data_batch['image'], data_batch['label']
        input_var = Variable(img.type('torch.FloatTensor'))
        label_var = Variable(label.type('torch.FloatTensor'))
        if args.gpus is not None:
            input_var = input_var.cuda()
            label_var = label_var.cuda()
        out = model(input_var)
        loss= criterion(out.view(-1, args.num_class), label_var.view(-1, args.num_class))
        loss.backward()
        optimizer.step() # We have accumulated enought gradients
        optimizer.zero_grad()
        if args.gpus is not None:
           targets.append(label_var.squeeze().data.cpu().numpy())
        else:
            targets.append(label_var.squeeze().data.numpy())
        # measure elapsed time
        batch_time.update(time.time() - end)
        losses.update(loss.item(), input_var.size(0))
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.6f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    .format( epoch, i, len(dataloader), batch_time=batch_time,
                        data_time=data_time, loss=losses, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            log.write(output + '\n')
            log.flush()
        #if i>10:
            #break

def validate(dataloader, model, criterion, iter, log): 
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    targets, preds = [], []
    for i, data_batch in enumerate(dataloader):
        img , label  = data_batch['image'], data_batch['label']
        with torch.no_grad():
            input_var = Variable(img.type('torch.FloatTensor'))
            label_var = Variable(label.type('torch.FloatTensor'))
            if args.gpus is not None:
                input_var = input_var.cuda()
                label_var = label_var.cuda()
        output = model(input_var)
        output = torch.sigmoid(output)
        if args.gpus is not None:
            targets.append(label_var.data.cpu().numpy())
            ################# Where the threshold is defined ###################
            preds.append((output>0.5).data.cpu().numpy().astype(np.int))
            ################# Where the threshold is defined ###################
        else:
            targets.append(label_var.data.numpy())
            preds.append((output>0.5).datanumpy().astype(np.int))
        loss = criterion(output.squeeze(), label_var.squeeze())  
        losses.update(loss.item(), input_var.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   i, len(dataloader), batch_time=batch_time, loss=losses))
            print(output)
            log.write(output + '\n')
            log.flush()
        torch.cuda.empty_cache()
        #if i>10:
            #break
    targets, preds = np.concatenate([array for array in targets], axis=0), np.concatenate([array for array in preds], axis=0)
    f1_score = averaged_f1_score(preds.reshape(-1, args.num_class), targets.reshape(-1, args.num_class))
    output = ' Validation : [{0}][{1}], F1 score: {f1_score:.4f} , loss:{loss:.4f}'.format( i, len(dataloader),
        f1_score = f1_score, loss = losses.avg) 
    print(output)
    log.write(output + '\n')
    log.flush() 
    return loss, f1_score

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.5 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
        param_group['weight_decay'] = decay         
def save_checkpoint(state, is_best_loss, is_best_acc, filename='fold'):
    torch.save(state, '%s/%s/%s_checkpoint.pth.tar' % (args.save_root, args.root_model, filename))
    if is_best_loss:
        shutil.copyfile('%s/%s/%s_checkpoint.pth.tar' % (args.save_root, args.root_model, filename),
                        '%s/%s/%s_best_loss.pth.tar' % (args.save_root, args.root_model, filename)) 
        print("checkpoint saved to",  '%s/%s/%s_best_loss.pth.tar' % (args.save_root, args.root_model, filename))           
    if is_best_acc:
        shutil.copyfile('%s/%s/%s_checkpoint.pth.tar' % (args.save_root, args.root_model, filename),
                        '%s/%s/%s_best_acc.pth.tar' % (args.save_root, args.root_model, filename)) 
        print("checkpoint saved to",  '%s/%s/%s_best_acc.pth.tar' % (args.save_root, args.root_model, filename))        
        
def main():
    USER_DATA_ROOT = '../user_data'
    if len(args.store_name)==0:
        args.store_name = '_'.join( ['optim:{}'.format(args.optim),
                                     'batch_size:{}'.format(args.batch_size), 
                                      'hidden_units:{}'.format(args.hidden_units)]) 
    setattr(args, 'save_root', os.path.join(USER_DATA_ROOT, args.store_name))
    print("save experiment to :{}".format(args.save_root))
    check_rootfolders()
    num_class = 53
    setattr(args, 'num_class', num_class)
    pos_weight = torch.ones([num_class])*25 # here we approximately set the pos weight as 25
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)         
    ###########################  Create the classifier ###################       
    model = MLP_RNN(args.hidden_units, args.num_class)  
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Params: {}".format(pytorch_total_params))
    if args.gpus is not None:
        if len(args.gpus)!=1:
            model = nn.DataParallel(model)
        model.cuda()
        pos_weight = pos_weight.cuda()
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), 
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)  
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                args.lr,
                                weight_decay=args.weight_decay)  
    train_loader, val_loader = DataLoader(args.batch_size, lmdb_file = args.lmdb,
                              val_ratio = args.val_ratio, train_num_workers = args.workers, 
                              val_num_workers = args.workers).create_dataloaders()
    log = open(os.path.join(args.save_root, args.root_log, '{}.txt'.format(args.store_name)), 'w')
    best_loss = 1000
    val_accum_epochs = 0
    best_acc = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        train(train_loader, model, criterion, optimizer, epoch, log)
        torch.cuda.empty_cache() 
        if val_loader is None:
            # save every epoch model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, True, False, filename='MLP_RNN_{}'.format(epoch))
        elif (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss_val, acc_val = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log)
            is_best_loss = loss_val< best_loss
            best_loss = min(loss_val, best_loss)
            is_best_acc = acc_val > best_acc
            best_acc  = max(acc_val , best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, is_best_loss, is_best_acc, filename='MLP_RNN')
            if not is_best_acc:
                val_accum_epochs+=1
            else:
                val_accum_epochs=0
            if val_accum_epochs>=args.early_stop:
                print("validation acc did not improve over {} epochs, stop".format(args.early_stop))
                break

if __name__=='__main__':
    main()
