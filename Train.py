import os
# Change the numbers when you want to train with specific gpus
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import torch
from ALSFNet import ALSFNet
import torch.nn.functional as F
from Utils.Datasets import get_data_loader
from Utils.Utils import make_numpy_img, inv_normalize_img, encode_onehot_to_mask, get_metrics, Logger
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from torch.optim.lr_scheduler import MultiStepLR
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


# class SegmentationLosses(object):
#     def __init__(self, ):
#         print("Ini")
def Weightloss(logit, target, weights):
    n, c, h, w = logit.size()
    # Calculate log probabilities
    logits_log_softmax = F.log_softmax(logit, dim=1).float()
    index = target.view(n, 1, h, w).long()
    #logits_log_probs = logits_log_softmax.gather(dim=1, index=target.view(n, 1, h, w).long())  # n, 1, h, w
    logits_log_probs = torch.take(logits_log_softmax, index)
    # # Multiply by exp(weights) [ weights on scale of 0-1, but taking exponent gives 1-e]
    # if weights is None:
    #     weights = torch.zeros_like(logits_log_probs)
    # else:
    #     weights = weights.unsqueeze(1)  # weights arrive as n, h, w
    weights = weights.unsqueeze(1)  # weights arrive as n, h, w

    weights_exp = torch.exp(weights) ** 2  # [0 - 1] --> [1 e**3=20]
    # print(torch.unique(weights_exp))
    assert weights_exp.size() == logits_log_probs.size()
    logits_weighted_log_probs = (logits_log_probs * weights_exp).view(n, -1)

    # Rescale the weights so loss is in approximately the same interval (distribution of weights may have a lot of variance)
    weighted_loss = logits_weighted_log_probs.sum(1) / weights_exp.view(n, -1).sum(1)
    loss = -1 * weighted_loss.mean()
    # Return mini-batch mean
    return loss

def TI_Loss(logit, target):
    logit = logit.exp()
    assert len(logit.shape) == 4
    assert len(target.shape) == 3
    beta = 0.8
    eps = 0.0001
    encoded_target = logit.detach() * 0
    target = target.long()
    encoded_target.scatter_(1, target.unsqueeze(1), 1)

    a = logit * encoded_target
    numerator = a.sum(0).sum(1).sum(1)
    b = (beta) * (encoded_target -a ).sum(0).sum(1).sum(1)
    c = (1-beta) * (logit -a ).sum(0).sum(1).sum(1)
    denominator = numerator + b + c + eps
    loss_per_channel = (1 - (numerator / denominator))
    return loss_per_channel.sum() / logit.size(1)

if __name__ == '__main__':
    model_infos = {
        # vgg16_bn, resnet50, resnet18, drn-c-42
        'backbone': 'drn-c-42',
        'pretrained': True,
        'out_keys': ['block4'],
        'in_channel': 3,
        'n_classes': 2,
        'top_k_s': 64,
        'top_k_c': 16,
        'encoder_pos': True,
        'decoder_pos': True,
        'model_pattern': ['L', 'X', 'A', 'S', 'C'],
        #'model_pattern': ['A', 'S', 'C'],
        'model': 'bridgebeta1',
        'BATCH_SIZE':  16,
        'IS_SHUFFLE': True,
        'NUM_WORKERS': 0,
        'DATASET': 'Tools/generate_dep_info/bridge_train_data.csv',
        'model_path': 'Checkpoints',
        'log_path': 'Results',
        # if you need the validation process.
        'IS_VAL': True,
        'VAL_BATCH_SIZE': 16,
        'VAL_DATASET': 'Tools/generate_dep_info/bridge_val_data.csv',
        # if you need the test process.
        'IS_TEST': True,
        'TEST_DATASET': 'Tools/generate_dep_info/bridge_test_data.csv',
        'IMG_SIZE': [512, 512],
        'PHASE': 'seg',

        ### bridge
        'PRIOR_MEAN': [0.3262195323495309, 0.3609351856879943, 0.3377596726760367],
        'PRIOR_STD': [0.0281896770612446, 0.025861062939073396, 0.025340187748247943],
        ###building
        #'PRIOR_MEAN': [0.5630680603779261, 0.589907237398366, 0.5252731729316505],
        #'PRIOR_STD': [0.04290568278635605, 0.03080307088813702, 0.03898509118242543],

        # if you want to load state dict
        # 'load_checkpoint_path': '',
        'load_checkpoint_path': r'E:\BuildingExtractionDataset\INRIA_ckpt_latest.pt',
        # if you want to resume a checkpoint
        'resume_checkpoint_path': '',

    }
    os.makedirs(model_infos['model_path'], exist_ok=True)
    if model_infos['IS_VAL']:
        os.makedirs(model_infos['log_path']+'/val', exist_ok=True)
    if model_infos['IS_TEST']:
        os.makedirs(model_infos['log_path']+'/test', exist_ok=True)
    logger = Logger(model_infos['log_path'] + '/log.log')

    data_loaders = get_data_loader(model_infos)
    model = ALSFNet(**model_infos)

    epoch_start = 0
    if model_infos['load_checkpoint_path'] is not None and os.path.exists(model_infos['load_checkpoint_path']):
        logger.write(f'load checkpoint from {model_infos["load_checkpoint_path"]}\n')
        state_dict = torch.load(model_infos['load_checkpoint_path'], map_location='cpu')
        model_dict = state_dict['model_state_dict']
        try:
            model_dict = OrderedDict({k.replace('module.', ''): v for k, v in model_dict.items()})
            model.load_state_dict(model_dict, strict=False)
        except Exception as e:
            model.load_state_dict(model_dict, strict=False)
    if model_infos['resume_checkpoint_path'] is not None and os.path.exists(model_infos['resume_checkpoint_path']):
        logger.write(f'resume checkpoint path from {model_infos["resume_checkpoint_path"]}\n')
        state_dict = torch.load(model_infos['resume_checkpoint_path'], map_location='cpu')
        epoch_start = state_dict['epoch_id']
        model_dict = state_dict['model_state_dict']
        logger.write(f'resume checkpoint from epoch {epoch_start}\n')
        try:
            model_dict = OrderedDict({k.replace('module.', ''): v for k, v in model_dict.items()})
            model.load_state_dict(model_dict)
        except Exception as e:
            model.load_state_dict(model_dict)
    model = model.cuda()
    device_ids = range(torch.cuda.device_count())
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.write(f'Use GPUs: {device_ids}\n')
    else:
        logger.write(f'Use GPUs: 1\n')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    max_epoch = 30
    scheduler = MultiStepLR(optimizer, [int(max_epoch*2/3), int(max_epoch*5/6)], 0.5)

    for epoch_id in range(epoch_start, max_epoch):
        pattern = 'train'
        model.train()  # Set model to training mode
        for batch_id, batch in enumerate(data_loaders[pattern]):
            # Get data
            img_batch = batch['img'].cuda()
            label_batch = batch['label'].cuda()
            weight_batch = batch['weight'].cuda()
            labelloss_batch = batch['labelloss'].cuda()
            # inference
            optimizer.zero_grad()
            logits, att_branch_output = model(img_batch)
            labelloss_batch[labelloss_batch == 255] = 1
            #print("labelloss_batch min:", labelloss_batch.min().item(), "labelloss_batch max:", labelloss_batch.max().item())

            # compute loss
            label_downs = F.interpolate(label_batch, att_branch_output.size()[2:], mode='nearest')
            loss_weight = Weightloss(logits, labelloss_batch, weight_batch).detach()
            loss_TI = TI_Loss(logits, labelloss_batch)
            loss =loss_TI + loss_weight
            # loss backward
            loss.backward()
            optimizer.step()

            if batch_id % 20 == 1:
                logger.write(
                    f'{pattern}: {epoch_id}/{max_epoch} {batch_id}/{len(data_loaders[pattern])} loss: {loss.item():.4f}\n')

        scheduler.step()
        patterns = ['val', 'test']
        for pattern_id, is_pattern in enumerate([model_infos['IS_VAL'], model_infos['IS_TEST']]):
            if is_pattern:
                # pred: logits, tensor, nBatch * nClass * W * H
                # target: labels, tensor, nBatch * nClass * W * H
                # output, batch['label']
                collect_result = {'pred': [], 'target': []}
                pattern = patterns[pattern_id]
                model.eval()
                for batch_id, batch in enumerate(data_loaders[pattern]):
                    # Get data
                    img_batch = batch['img'].cuda()
                    label_batch = batch['label'].cuda()
                    weight_batch = batch['weight'].cuda()
                    labelloss_batch = batch['labelloss'].cuda()
                    img_names = batch['img_name']
                    collect_result['target'].append(label_batch.data.cpu())

                    # inference
                    with torch.no_grad():
                        logits, att_branch_output = model(img_batch)

                    collect_result['pred'].append(logits.data.cpu())
                    # get segmentation result, when the phase is test.
                    pred_label = torch.argmax(logits, 1)
                    pred_label *= 255

                    if pattern == 'test' or batch_id % 5 == 1:
                        batch_size = pred_label.size(0)
                        # k = np.clip(int(0.3 * batch_size), a_min=1, a_max=batch_size)
                        # ids = np.random.choice(range(batch_size), k, replace=False)
                        ids = range(batch_size)
                        for img_id in ids:
                            img = img_batch[img_id].detach().cpu()
                            target = label_batch[img_id].detach().cpu()
                            pred = pred_label[img_id].detach().cpu()
                            img_name = img_names[img_id]

                            img = make_numpy_img(
                                inv_normalize_img(img, model_infos['PRIOR_MEAN'], model_infos['PRIOR_STD']))
                            target = make_numpy_img(encode_onehot_to_mask(target)) * 255
                            pred = make_numpy_img(pred)

                            #vis = np.concatenate([img / 255., target / 255., pred / 255.], axis=1)
                            #vis = np.clip(vis, a_min=0, a_max=1)
                            pred = pred / 255.
                            pred = np.clip(pred, a_min=0, a_max=1)
                            file_name = os.path.join(model_infos['log_path'], pattern, f'Epoch_{epoch_id}_{img_name.split(".")[0]}.jpg')
                            plt.imsave(file_name, pred)

                collect_result['pred'] = torch.cat(collect_result['pred'], dim=0)
                collect_result['target'] = torch.cat(collect_result['target'], dim=0)
                mIoU, precision, recall, F1_score = get_metrics('seg', **collect_result)
                logger.write(f'{pattern}: {epoch_id}/{max_epoch} mIou:{mIoU:.4f} precision:{precision[-1]:.4f} recall:{recall[-1]:.4f} F1:{F1_score[-1]:.4f}\n')
        #if epoch_id % 20 == 1:
        #    torch.save({
        #        'epoch_id': epoch_id,
        #        'model_state_dict': model.state_dict()
        #    }, os.path.join(model_infos['model_path'], f'ckpt_{epoch_id}.pt'))
        torch.save({
            'epoch_id': epoch_id,
            'model_state_dict': model.state_dict()
        }, os.path.join(model_infos['model_path'], model_infos['model'] + f'ckpt_latest.pt'))

