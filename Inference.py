from ViT import *
import numpy as np
import itertools
import time
from parameters import *
import torch
from Accdataloader import *
from utils import *
import timm
from timm.loss import LabelSmoothingCrossEntropy
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import logging
from torch.utils.tensorboard import SummaryWriter

if (opt.OptimalQ):
    writer = SummaryWriter('./output/optimal')
else:
    writer = SummaryWriter('./output/linear_' + str(int(10 * opt.strength)))

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

transform_train = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def test():
    if (opt.scale == 'tiny'):
        Layers = 12
        HiddenSize = 192
        Heads = 3
        MLPSize = 768  # MLP ratio 4
    elif (opt.scale == 'small'):
        Layers = 12
        HiddenSize = 384
        Heads = 12
        MLPSize = 1536  # MLP ratio 4
    elif (opt.scale == 'base'):
        Layers = 12
        HiddenSize = 768
        Heads = 12
        MLPSize = 3072
    elif (opt.scale == 'large'):
        Layers = 24
        HiddenSize = 1024
        Heads = 16
        MLPSize = 4096
    elif (opt.scale == 'huge'):
        Layers = 32
        HiddenSize = 1280
        Heads = 16
        MLPSize = 5120

    noise_vit = NoiseViT(patch_size=opt.patch_size, num_classes=opt.num_classes, embed_dim=HiddenSize, depth=Layers,
                         num_heads=Heads)
    noise_vit.to(device)

    # load saved model
    noise_vit.load_state_dict(torch.load(opt.model_saved_path))


    print('model parameters:', sum(param.numel() for param in noise_vit.parameters()) / 1e6)
    te_dataloader = get_test_loader(opt.test_path)  # give test images folder as --imagenet_path


    start_t = time.time()

    loss_fn = LabelSmoothingCrossEntropy(0.1)

    #############################################################################
    batch_time = AverageMeter()
    accs = AverageMeter()
    epoch_te_loss = AverageMeter()
    accs_top5 = AverageMeter()

    total = 0
    correct = 0
    correct_top5 = 0
    for i, (te_transformed_normalized_img, te_labels) in enumerate(te_dataloader):
        te_transformed_normalized_img = te_transformed_normalized_img.float().cuda()

        with torch.no_grad():
            noise_vit.eval()
            outputs = noise_vit(te_transformed_normalized_img)
            cls_loss = loss_fn((outputs), (te_labels.to(device)))

            # acc---------------------------------------------------------------------
            _, predictions = torch.max(outputs, 1)
            #print(predictions)
            total += (te_labels).size(0)
            correct += ((predictions) == (te_labels.to(device))).sum().item()  # .cpu()
            # ------------------------------------------------------------------------
            epoch_te_loss.update(cls_loss.detach())
            accs.update(correct / total)  # acc = correct/total

            # Get top 5 predictions
            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)

            # Check if te_labels are in the top 5 predictions
            correct_top5 += (top5_pred == te_labels.view(-1, 1).to(device)).sum().item()
            accs_top5.update(correct_top5 / total)  # acc = correct/total
            batch_time.update(time.time() - start_t)
        # Print log info
        # print('te batch size', len(te_labels))
        if i % 100 == 0:  # opt.log_step
            # print('======================== print results \t' + time.asctime(time.localtime(time.time())) + '=============================')
            print('Time(s) from start {batch_time.val:.3f} \t'
                  'Classification_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(batch_time=batch_time,
                                                                            loss=epoch_te_loss,
                                                                            top1=accs,
                                                                            top5=accs_top5))
    writer.add_scalar('clssification_test_loss', epoch_te_loss.avg)
    writer.add_scalar('test_acc', accs.avg)  # add_scalar
    writer.add_scalar('test_top5 acc', accs_top5.avg)  # add_scalar


    writer.close()


if __name__ == '__main__':
    set_seed(42)

    test()



