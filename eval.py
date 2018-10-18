import torch
import numpy as np
from utils import *
import os
import json

def language_eval(dataset, preds, split):
    annFile = 'F:/mscoco/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/' + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, split, opt):
    model.eval()
    loader.reset_iterator(split)
    beam_size = opt.beam_size

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + opt.batch_size

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp

        with torch.no_grad():
            loss = crit(model('forward', fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:]).item()
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        tmp = [data['fc_feats'][np.arange(opt.batch_size) * opt.seq_per_img],
               data['att_feats'][np.arange(opt.batch_size) * opt.seq_per_img],
               data['att_masks'][np.arange(opt.batch_size) * opt.seq_per_img] if data['att_masks'] is not None else None]

        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        # greedy search
        with torch.no_grad():
            seq = model('sample',fc_feats, att_feats, None,att_masks)[0].data

        sents = decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if opt.val_num_images != -1:
            ix1 = min(ix1, opt.val_num_images)

        for i in range(n - ix1):
            predictions.pop()
        if n+1 % opt.print_eval_every == 0:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))
        if data['bounds']['wrapped']:
            break
        if opt.val_num_images >= 0 and n >= opt.val_num_images:
            break

    lang_stats = None
    lang_stats = language_eval('coco', predictions, split)

    model.train()
    return loss_sum/loss_evals, predictions, lang_stats
