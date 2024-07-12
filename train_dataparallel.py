import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DRIVPokcet_net import DRIVPocket
from tqdm import tqdm
from torch.optim import lr_scheduler
import numpy as np
import logging
import argparse
import wandb
import sys
import os
from Dataset import Protein_Dataset
import molgrid
from modules.dice_loss import dice_loss
from modules.sophia import SophiaG
from modules.riconv2_utils import compute_LRA

torch.backends.cudnn.benchmark = True

box_size = 65


def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('--train_types', type=str, required=False,
                        help="training types file", default='./scPDB_center_info/seg_scPDB_train0_96.types')
    parser.add_argument('--test_types', type=str, required=False,
                        help="test types file", default='./scPDB_center_info/seg_scPDB_test0_96.types')
    parser.add_argument('-d', '--data_dir', type=str, required=True,
                        help="Root directory of data", default="./Datasets/ScPDB/scPDB")
    parser.add_argument('--train_recmolcache', type=str, required=True,
                        help="path to train receptor molcache",
                        default="./Datasets/scPDB_new.molcache2")
    parser.add_argument('--test_recmolcache', type=str, required=True,
                        help="path to test receptor molcache",
                        default="./Datasets/scPDB_new.molcache2")
    parser.add_argument('-e', '--num_epochs', type=int, required=False,
                        help="Number of epochs", default=150)
    parser.add_argument('-b', '--batch_size', type=int, required=False,
                        help="Batch size for training, default 5", default=5)
    parser.add_argument('--num_classes', type=int, required=False,
                        help="Output channels for predicted masks, default 1", default=1)
    parser.add_argument('-s', '--seed', type=int, required=False, help="Random seed, default 0", default=1)
    parser.add_argument('-r', '--run_name', type=str, help="name for wandb run", required=False)
    parser.add_argument('-o', '--outprefix', type=str, help="Prefix for output files", required=False,
                        default='./models/seg')
    parser.add_argument('--checkpoint', type=str, required=False, help="file to continue training from")
    parser.add_argument('--solver', type=str, help="Solver type. Default is SGD, Nesterov or Adam", default='SGD')
    parser.add_argument('--step_reduce', type=float,
                        help="Reduce the learning rate by this factor with dynamic stepping, default 0.1",
                        default=0.1)
    parser.add_argument('--step_end_cnt', type=float, help='Terminate training after this many lr reductions',
                        default=3)
    parser.add_argument('--step_when', type=int,
                        help="Perform a dynamic step (reduce base_lr) when training has not improved after these many epochs, default 2",
                        default=2)
    parser.add_argument('--base_lr', type=float, help='Initial learning rate, default 0.001', default=0.001)
    parser.add_argument('--momentum', type=float, help="Momentum parameters, default 0.9", default=0.9)
    parser.add_argument('--weight_decay', type=float, help="Weight decay, default 0.001", default=0.001)
    parser.add_argument('--clip_gradients', type=float, default=10.0, help="Clip gradients threshold (default 10)")
    parser.add_argument('--point_num', type=int, default=256, help="number of ponitcloud from knn")
    parser.add_argument('--syncBN', type=bool, default=True)
    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--usebaseline', default=0, type=int, help='Unet')
    parser.add_argument('--usewandb', default=1, type=int, help='use wandb or not')
    parser.add_argument('--ite', type=int, default=2)
    parser.add_argument('--is_mask', type=int, default=0)
    parser.add_argument('--is_debug', type=int, default=0)
    parser.add_argument('--DATA_ROOT', type=str, default='/home/dbw/PythonProject/DeepPocket_pp/coach420',
                        help="the path of data root")

    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--mask_dist', type=float, default=3.5)
    parser.add_argument('--is_dvo', type=int, default=1)
    parser.add_argument('--is_dca', type=int, default=0)
    parser.add_argument('-n', '--top_n', type=int, default=0)
    parser.add_argument('--test_set', type=str, default='coach420')  # coach420,sc6k,holo4k,pdbbind,apoholo
    parser.add_argument('--model_path', type=str, default='')
    args = parser.parse_args(argv)

    argdict = vars(args)
    line = ''
    for (name, val) in list(argdict.items()):
        if val != parser.get_default(name):
            line += ' --%s=%s' % (name, val)

    return (args, line)


def cal_dice_coeff(input, target):
    eps = 0.000001
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target) + eps
    t = (2 * inter.float() + eps) / union.float()
    return t


def cal_IOU(input, target):
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target)
    t = (inter.float()) / (union.float() - inter.float())
    return t


def Dice_loss(input, target):
    dice_loss_ = cal_dice_coeff(input, target)
    return 1 - dice_loss_


def knn(Points, xyz, num=200):
    ref_c = torch.stack([xyz] * Points.shape[0], dim=0)
    query_c = Points
    delta = query_c - ref_c
    distances = torch.sqrt(torch.pow(delta, 2).sum(dim=1))
    sorted_dist, indices = torch.sort(distances)
    return query_c[indices[:num]]


def pc_normalize(pc):
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))
    pc = pc / m
    return pc


def get_model_gmaker_eproviders(args):
    # gridmaker with defaults
    gmaker_img = molgrid.GridMaker(dimension=31.5)
    # grid maker for ground truth tensor
    gmaker_mask = molgrid.GridMaker(dimension=31.5, binary=True, gaussian_radius_multiple=-1, resolution=0.5)

    return gmaker_img, gmaker_mask


def get_edge(coordinateset, binary_labels, center):
    # Create ground truth tensor
    # 假设你有两个张量，xyz_coordinates 和 binary_labels
    # xyz_coordinates 是一个形状为 (8, N, 3) 的张量，其中 N 是每个集合的点的数量，3 表示 XYZ 坐标的维度
    # binary_labels 是一个形状为 (8, N) 的张量，包含每个集合中点的二进制标签
    gmaker_img = molgrid.GridMaker(dimension=32, gaussian_radius_multiple=3)
    grid_nps = []
    # 输出结果
    for i, points in enumerate(coordinateset):
        marked_as_1_indices = torch.nonzero(binary_labels[i] == 1, as_tuple=True)
        points_marked_as_1 = points[marked_as_1_indices[0]]
        # print(f"集合 {i + 1} 中坐标集合标记为1的点：", points_marked_as_1)
        c2grid = molgrid.Coords2Grid(gmaker_img, center=molgrid.float3(float(center[0][i]), float(center[1][i]),
                                                                       float(center[2][i])))
        origtypes = torch.ones(points_marked_as_1.numpy().shape[0], 1)
        radii = torch.ones((points_marked_as_1.numpy().shape[0])) * 3
        grid_gen = c2grid(points_marked_as_1, origtypes, radii)
        grid_np = grid_gen.numpy()
        # grid_np = binary_dilation(grid_np[0], cube(3))
        grid_np = grid_np.astype(float)
        if len(center) > 1:
            grid_nps.append(grid_np)
        else:
            grid_nps.append(grid_np[0])
    grid_nps = torch.tensor(np.array(grid_nps))
    mask = grid_nps >= 1.2
    grid_nps[mask] = 0
    return grid_nps


def Shuffle(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 使用random.shuffle打乱行的顺序
    random.shuffle(lines)

    # 将打乱后的行写入新的文件
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(lines)
    return output_file


def initialize_model(model, args):
    def weights_init(m):
        '''initialize model weights with xavier'''
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            torch.nn.init.kaiming_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

        if isinstance(m, nn.BatchNorm3d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.cuda()
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.apply(weights_init)


def train(gmaker_img, gmaker_mask, args, device, total=None):
    checkpoint = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    args.base_lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增

    train_ProteinDataset = Protein_Dataset.TrainScPDB(args.train_types, data_root=args.data_dir,
                                                      recmolcache=args.train_recmolcache, cache_structs=True,
                                                      shuffle=True, stratify_receptor=False, balanced=False, )
    train_ProteinDataset.set_transform(True)
    test_ProteinDataset = Protein_Dataset.TrainScPDB(args.test_types, data_root=args.data_dir,
                                                     recmolcache=args.train_recmolcache, cache_structs=True)
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    test_loader = torch.utils.data.DataLoader(test_ProteinDataset,
                                              batch_size=batch_size,
                                              pin_memory=True)

    model = DRIVPocket(in_channels=14,
                     out_channels=args.num_classes,
                     feature_size=32,
                     num_heads=4,
                     hidden_size=512,
                     depths=[1, 1, 1, 1],
                     dims=[64, 128, 256, 512],
                     )
    model = nn.DataParallel(model)
    model.to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    if args.freeze:
        print("----freeze----")
        for name, param in model.named_parameters():
            if "RIconv2_Unter" not in name:
                # 冻结权重
                param.requires_grad = False
    if args.usewandb:
        wandb.init(project='deep-pocket-pp', name=args.run_name)
        wandb.watch(model)
    num_epochs = args.num_epochs
    outprefix = args.outprefix
    prev_total_loss_snap = ''
    prev_total_dice_snap = ''
    prev_total_IOU_snap = ''
    prev_snap = ''
    initial = 0
    box_size = 65
    last_test = 0
    iterations = 5

    if args.checkpoint:
        if 'Epoch' in checkpoint:
            initial = checkpoint['Epoch']

    if 'SGD' in args.solver:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif 'Nesterov' in args.solver:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    elif 'Adam' in args.solver:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    elif 'AdamW' in args.solver:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, betas=(0.95, 0.99),
                                      weight_decay=args.weight_decay)
    else:
        print("No test solver argument passed (SGD, Adam, Nesterov)")
        sys.exit(1)

    if args.checkpoint:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = lr_scheduler.StepLR(optimizer, step_size=iterations, gamma=0.9)
    if args.checkpoint:
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    Bests = {}
    Bests['train_epoch'] = 0
    Bests['test_loss'] = torch.from_numpy(np.asarray(np.inf))
    Bests['test_accuracy'] = torch.from_numpy(np.asarray([0]))
    Bests['dice_coeff'] = torch.from_numpy(np.asarray([0]))
    Bests['IOU'] = torch.from_numpy(np.asarray([0]))
    if args.checkpoint:
        if 'Bests' in checkpoint:
            Bests = checkpoint['Bests']
    criterion = nn.BCELoss()
    Bceloss = nn.BCELoss()
    logging.info("Started Training.....")
    for epoch in range(initial, num_epochs):
        model.train()
        Shuffle(args.train_types, "/home/dbw/PythonProject/DeepPocket_pp/dataset_temp/epoch{}.txt".format(epoch))
        train_ProteinDataset = Protein_Dataset.TrainScPDB(
            "/home/dbw/PythonProject/DeepPocket_pp/dataset_temp/epoch{}.txt".format(epoch),
            data_root=args.data_dir,
            recmolcache=args.train_recmolcache, cache_structs=True,
            shuffle=True, stratify_receptor=False, balanced=False, )
        train_ProteinDataset.set_transform(True)
        train_loader = torch.utils.data.DataLoader(train_ProteinDataset,
                                                   batch_size=batch_size,
                                                   pin_memory=True)
        with tqdm(train_loader, desc="epoch:{}".format(epoch)) as train_bar:
            for batch in train_bar:
                # extract labels and centers of batch datapoints
                input_tensor, riconv_tensor, mask_tensor, labels, protein_name, feature, edge_pocket, protein_coords = batch
                mask_tensor, input_tensor, edge_pocket = mask_tensor.float().to(device), input_tensor.to(
                    device), edge_pocket.to(device)
                optimizer.zero_grad()
                # Take only the first 14 channels as that is for proteins, other 14 are ligands and will remain 0.
                if args.usebaseline == 1:
                    masks_pred = model(input_tensor[:, :14])
                else:
                    norm = compute_LRA(riconv_tensor, False)
                    points = torch.cat([riconv_tensor, norm], dim=-1)
                    point_pred, masks_pred = model(input_tensor[:, :14], points, feature)
                # running_loss += loss.item()
                predictions = torch.where(masks_pred > 0.5, 1, 0)
                acc = torch.mean(
                    (mask_tensor == predictions).float())  # Pixel-wise accuracy can be misleading in case of class
                loss_dice = dice_loss(masks_pred, mask_tensor)
                loss_point = Bceloss(point_pred, edge_pocket)
                pred = (masks_pred > 0.5).float()

                dice = cal_dice_coeff(pred, mask_tensor)
                IOU = cal_IOU(pred, mask_tensor)
                loss_bce = criterion(masks_pred, mask_tensor)
                Loss =  0.5 * loss_dice + 2 * loss_bce + 1 * loss_point
                Loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradients)
                optimizer.step()
                # print(IOU)
                if args.usewandb:
                    wandb.log({'train_loss': Loss.item(), 'loss_point': loss_point, 'loss_bce': loss_bce,
                               'train_accuracy': acc, 'train_dice': dice, 'train_IOU': IOU,
                               'learning rate': optimizer.param_groups[0]['lr']})

        torch.cuda.empty_cache()
        test_loss, test_acc, dice_coeff, IOU, total_fuse_dice, tatal_fuse_IOU, point_loss = test(model, test_loader,
                                                                                                 gmaker_img,
                                                                                                 gmaker_mask, args,
                                                                                                 criterion,
                                                                                                 device)
        scheduler.step()
        if test_loss < Bests['test_loss']:
            Bests['test_loss'] = test_loss
            if args.usewandb:
                wandb.run.summary["test_loss"] = Bests['test_loss']
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'Bests': Bests,
                        'Epoch': epoch + 1}, outprefix + '_best_test_loss_' + str(epoch + 1) + '.pth.tar')
            if prev_total_loss_snap:
                os.remove(prev_total_loss_snap)
            prev_total_loss_snap = outprefix + '_best_test_loss_' + str(epoch + 1) + '.pth.tar'
        if IOU > Bests['IOU']:
            Bests['IOU'] = IOU
            if args.usewandb:
                wandb.run.summary["IOU"] = Bests['IOU']
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'Bests': Bests,
                        'Epoch': epoch + 1}, outprefix + '_best_test_IOU_' + str(epoch + 1) + '.pth.tar')
            if prev_total_IOU_snap:
                os.remove(prev_total_IOU_snap)
            prev_total_IOU_snap = outprefix + '_best_test_IOU_' + str(epoch + 1) + '.pth.tar'
        if dice_coeff > Bests['dice_coeff']:
            Bests['dice_coeff'] = dice_coeff
            if args.usewandb:
                wandb.run.summary["dice_coeff"] = Bests['dice_coeff']
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'Bests': Bests,
                        'Epoch': epoch + 1}, outprefix + '_best_dice_coeff_' + str(epoch + 1) + '.pth.tar')
            if prev_total_dice_snap:
                os.remove(prev_total_dice_snap)
            prev_total_dice_snap = outprefix + '_best_dice_coeff_' + str(epoch + 1) + '.pth.tar'
            Bests['train_epoch'] = epoch
        if epoch - Bests['train_epoch'] >= args.step_when and optimizer.param_groups[0]['lr'] <= (
                (args.step_reduce) ** args.step_end_cnt) * args.base_lr:
            last_test = 1
        print(
            "Epoch {}, total_test_loss: {:.3f},total_test_accuracy: {:.3f},total_dice_coeff: {:.3f}, Best_test_loss: {:.3f},Best_dice_coeff: {:.3f},learning_Rate: {:.7f}，IOU:{:.3f}, total_fuse_dice:{:.3f},point_loss:{:.3f}".format(
                epoch + 1, test_loss, test_acc, dice_coeff, Bests['test_loss'],
                Bests['dice_coeff'], optimizer.param_groups[0]['lr'], IOU, total_fuse_dice, point_loss))
        if args.usewandb:
            # wandb.log({'test_loss': test_loss, 'test_accuracy': test_acc, 'dice_coeff': dice_coeff,
            #            'learning rate': optimizer.param_groups[0]['lr'],'total_fuse_dice': total_fuse_dice,'tatal_fuse_IOU':tatal_fuse_IOU})
            wandb.log({'test_loss': test_loss, 'test_accuracy': test_acc, 'dice_coeff': dice_coeff,
                       'learning rate': optimizer.param_groups[0]['lr'], "point_test_loss：": point_loss})
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'Bests': Bests,
                    'Epoch': epoch + 1}, outprefix + '_' + str(epoch + 1) + '.pth.tar')
        if prev_snap:
            os.remove(prev_snap)
        prev_snap = outprefix + '_' + str(epoch + 1) + '.pth.tar'
        if last_test:
            return Bests
        torch.cuda.empty_cache()
    logging.info("Finished Training")


def test(model, test_loader, gmaker_img, gmaker_mask, args, criterion, device):
    model.eval()
    torch.cuda.empty_cache()
    running_acc, fuse_IOU = 0.0, 0.0
    running_loss = 0.0
    tot_dice = 0.0
    tot_IOU = 0.0
    total_fuse_dice = 0.0
    total_point_loss = 0.0
    Bceloss = nn.BCELoss()

    for batch in test_loader:
        input_tensor, riconv_tensor, mask_tensor, labels, _, feature, edge_pocket, protein_coords = batch
        mask_tensor, input_tensor, edge_pocket = mask_tensor.float().to(device), input_tensor.to(
            device), edge_pocket.to(device)
        with torch.no_grad():
            if args.usebaseline == 1:
                masks_pred = model(input_tensor[:, :14])
            else:
                norm = compute_LRA(riconv_tensor, False)
                points = torch.cat([riconv_tensor, norm], dim=-1)
                point_pred, masks_pred = model(input_tensor[:, :14], points, feature)
        loss_point = Bceloss(point_pred, edge_pocket)
        loss = criterion(masks_pred, mask_tensor)
        _, predictions = torch.max(masks_pred, 1)
        running_loss += loss.detach().cpu()
        total_point_loss += loss_point.detach().cpu()
        running_acc += torch.mean((mask_tensor == predictions).float()).detach().cpu()
        pred = masks_pred
        pred = (pred > 0.5).float()
        tot_dice += cal_dice_coeff(pred, mask_tensor).detach().cpu()
        tot_IOU += cal_IOU(pred, mask_tensor).detach().cpu()
    test_loss = running_loss /len(test_loader)
    point_loss = total_point_loss / len(test_loader)
    test_acc = running_acc /len(test_loader)
    dice_coeff = tot_dice / len(test_loader)
    IOU = tot_IOU / len(test_loader)
    return test_loss, test_acc, dice_coeff, IOU, total_fuse_dice, fuse_IOU, point_loss


if __name__ == "__main__":
    (args, cmdline) = parse_args()
    gmaker_img, gmaker_mask = get_model_gmaker_eproviders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Bests = train(gmaker_img, gmaker_mask, args, device)
    print(Bests)
