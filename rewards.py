import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch
import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
# from pycocoevalcap.cider.cider import Cider as CiderD
from pycocoevalcap.bleu.bleu import Bleu

CiderD_scorer = None
Bleu_scorer = None

def init_scorer(cache_tokens):
    global CiderD_scorer
    if CiderD_scorer is None:
        CiderD_scorer = CiderD(df=cache_tokens)
    else:
        CiderD_scorer = CiderD_scorer
    # CiderD_scorer = CiderD_scorer or CiderD(df=cache_tokens)
    global Bleu_scorer
    if Bleu_scorer is None:
        Bleu_scorer = Bleu(4)
    else:
        Bleu_scorer = Bleu_scorer
    # Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(model, fc_feats, att_feats, att_masks, data, gen_result, opt):
    batch_size = gen_result.size(0) #50
    seq_per_img = batch_size // len(data['gts'])
    model.eval()
    with torch.no_grad():
        greedy_res, _ = model('sample', fc_feats, att_feats, att_masks)
    model.train()
    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()

    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[i + batch_size] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        #print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        #print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()
    # rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())
    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = -input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output






