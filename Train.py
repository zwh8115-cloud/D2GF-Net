import os
import random
import subprocess
import sys
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from lib.models.detectors.Net import Network
from utils.sdy_data_val import get_loader
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
from utils.loss_function import *
import logging
import torch.backends.cudnn as cudnn

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def slim_state_dict(model):
    return {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith('tsf.cgdf.dino.')
    }


def train(train_loader, model, optimizer, epoch, save_path, writer, cur_loss):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, edges, depths) in enumerate(
            train_loader,
            start=1,
        ):
            optimizer.zero_grad()

            images = images.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            edges = edges.cuda(non_blocking=True)
            depths = depths.cuda(non_blocking=True)

            preds = model(images, depths)

            loss_fg = (
                cur_loss(preds[0], gts) / 8
                + cur_loss(preds[1], gts) / 4
                + cur_loss(preds[2], gts) / 2
                + cur_loss(preds[3], gts)
            )
            loss_low = cur_loss(preds[4], gts)

            loss = loss_fg + loss_low

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss_fg: {:.4f} Loss_low: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_fg.data, loss_low.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'
                    'Loss_fg: {:0.4f} Loss_low: {:.4f}'.
                        format(epoch, opt.epoch, i, total_step, loss.data, loss_fg.data, loss_low.data))

        loss_all /= epoch_step
        slim = slim_state_dict(model)
        torch.save(
            {
                'model': slim,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            },
            save_path + 'Net_epoch_last.pth',
        )
        if epoch >= 50:
            torch.save(slim, save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(slim_state_dict(model), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=60, help='epoch number')
    parser.add_argument('--lr', type=float, default=2.0e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=12, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=448, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='resume start epoch (used with --load on plain weight files)')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str,
                        default='./data/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--train_depth_root', type=str,
                        default='./data/TrainDataset/Depth/',
                        help='the training depth maps root')
    parser.add_argument('--dinov2_path', type=str,
                        default='./pretrained/dinov2_large.safetensors',
                        help='path to the pretrained DINOv2 checkpoint (timm format)')
    parser.add_argument('--dinov2_model', type=str,
                        default='vit_large_patch14_dinov2',
                        help='DINOv2 timm model name')
    parser.add_argument('--dinov2_input_size', type=int, default=448,
                        help='input size used by the DINOv2 branch')
    parser.add_argument('--disable_dinov2', action='store_true',
                        help='disable the DINOv2 concat fusion branch')
    parser.add_argument('--save_path', type=str,
                        default='./checkpoints/D2GF-Net/',
                        help='the path to save model and log')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--loss_type', type=str, default='bei', help='the type of loss function')
    opt = parser.parse_args()

    if opt.loss_type == 'bei':
        cur_loss = hybrid_e_loss
    elif opt.loss_type == 'wce':
        cur_loss = wce_loss
    elif opt.loss_type == 'wiou':
        cur_loss = wiou_loss
    elif opt.loss_type == 'e':
        cur_loss = e_loss
    else:
        raise Exception('No Type Matching')

    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    model = Network(
        use_dinov2=not opt.disable_dinov2,
        dinov2_model=opt.dinov2_model,
        dinov2_path=opt.dinov2_path,
        dinov2_input_size=opt.dinov2_input_size,
    ).cuda()

    train_state = None
    if opt.load is not None:
        checkpoint = torch.load(opt.load, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint and 'optimizer' in checkpoint:
            train_state = checkpoint
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict,
            strict=False,
        )
        print(
            'checkpoint missing keys: {}, unexpected keys: {}'.format(
                len(missing_keys),
                len(unexpected_keys),
            )
        )
        print('load model from ', opt.load)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    params = model.parameters()
    print('model paramters', sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info("model paramters:" + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print('load data...')
    data_generator = torch.Generator()
    data_generator.manual_seed(SEED)
    train_loader = get_loader(image_root=os.path.join(opt.train_root, 'Imgs'),
                              gt_root=os.path.join(opt.train_root, 'GT'),
                              grad_root=os.path.join(opt.train_root, 'Edges'),
                              depth_root=opt.train_depth_root,
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8,
                              worker_init_fn=seed_worker,
                              generator=data_generator)
    total_step = len(train_loader)

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    start_epoch = opt.start_epoch
    if train_state is not None:
        optimizer.load_state_dict(train_state['optimizer'])
        start_epoch = train_state.get('epoch', 0) + 1
        print('Resume training from epoch {}'.format(start_epoch))
        logging.info('Resume from {} at epoch {}'.format(opt.load, start_epoch))

    print("Start train...")
    for epoch in range(start_epoch, opt.epoch + 1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer, cur_loss)
    writer.close()

    last_ckpt = os.path.join(save_path, 'Net_epoch_{}.pth'.format(opt.epoch))
    if not os.path.isfile(last_ckpt):
        print('No final checkpoint found, skip auto test.')
    else:
        print('Training finished, auto testing checkpoint: {}'.format(last_ckpt))
        del model, optimizer
        torch.cuda.empty_cache()
        train_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = [
            sys.executable, os.path.join(train_dir, 'Test.py'),
            '--pth_path', last_ckpt,
            '--testsize', str(opt.trainsize),
            '--gpu_id', opt.gpu_id,
            '--dinov2_model', opt.dinov2_model,
            '--dinov2_path', opt.dinov2_path,
            '--dinov2_input_size', str(opt.dinov2_input_size),
        ]
        if opt.disable_dinov2:
            cmd.append('--disable_dinov2')
        subprocess.run(cmd, check=True, cwd=train_dir)
