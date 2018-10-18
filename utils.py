import torch
import torch.nn as nn

def decode_sequence(ix_to_word, seq):
    # N is batch size of seq
    # D is length of seq
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

# def clip_gradient(optimizer, grad_clip):
#     param = optimizer['params']
#     param.grad.data.clamp_(-grad_clip, grad_clip)

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):

        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        target = torch.cuda.LongTensor(target.cpu().detach().numpy())
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output