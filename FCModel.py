import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    # implements beam search
    # calls beam_step and return the final set of beams
    # augments log-probabilities with diversity term when number of ground > 1

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):
        pass

class LSTMCore(nn.Module):
    def __init__(self, opt):
        super(LSTMCore, self).__init__()
        self.opt = opt
        self.i2h = nn.Linear(opt.embedding_size, 5 * opt.rnn_size)
        self.h2h = nn.Linear(opt.rnn_size, 5*opt.rnn_size)
        self.dropout = nn.Dropout(opt.drop_prob_lm)

    def forward(self, xt, state):
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.opt.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1,0,self.opt.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1,self.opt.rnn_size,self.opt.rnn_size)
        out_gate = sigmoid_chunk.narrow(1,self.opt.rnn_size*2,self.opt.rnn_size)

        in_transform = torch.max(\
            all_input_sums.narrow(1, 3 * self.opt.rnn_size, self.opt.rnn_size),
            all_input_sums.narrow(1, 4 * self.opt.rnn_size, self.opt.rnn_size))
        # f * s^t-1 + i * g
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        # increase dimension
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class FCModel(CaptionModel):
    def __init__(self, opt):
        super(FCModel, self).__init__()
        self.opt = opt

        self.img_embed = nn.Linear(opt.fc_feat_size, opt.embedding_size)
        #self.core = LSTMCore(opt)

        self.core = getattr(nn, opt.rnn_type.upper())(opt.embedding_size, opt.rnn_size, opt.num_layers,bias=False, dropout=opt.drop_prob_lm)

        self.embed = nn.Embedding(opt.vocab_size + 1, opt.embedding_size)
        self.logit = nn.Linear(opt.rnn_size, opt.vocab_size+1)

        self.dropout = nn.Dropout(opt.drop_prob_lm)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.opt.rnn_type == 'lstm':
            return (weight.new_zeros(self.opt.num_layers, bsz, self.opt.rnn_size),
                    weight.new_zeros(self.opt.num_layers, bsz, self.opt.rnn_size))
        else:
            return weight.new_zeros(bsz, self.opt.rnn_size)

    def sample(self, fc_feats, att_feats, att_masks):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.opt.seq_length, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.opt.seq_length)
        fc_feats = fc_feats.squeeze(1)
        # Greedy Search
        for i in range(self.opt.seq_length + 2):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                if i == 1:
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = self.embed(it)
            output, state = self.core(xt.unsqueeze(0), state)
            Logprobs = F.log_softmax(self.logit(output.squeeze(0)), dim=1)

            if i == self.opt.seq_length + 1:
                break
            if self.opt.sample_max:
                sampleLogprobs, it = torch.max(Logprobs.data, 1)
                it = it.view(-1).long()
            else:
                prob_prev = torch.exp(Logprobs.data).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = Logprobs.gather(1, it)
                it = it.view(-1).long()

            if i >= 1:
                # stop when all finished
                if i == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)

                seq[:,i-1] = it[:] # seq[t] the input of t+2 time step
                seqLogprobs[:,i-1] = sampleLogprobs.view(-1)
                if unfinished.sum() == 0:
                    break

        return seq, seqLogprobs


    def forward(self, mode, fc_feats, att_feats, seq, att_masks = None):
        # fc_feats [50, 1, 2048]
        # seq [50, 18]
        # vocab size 9487
        if mode == 'sample':
            return self.sample(fc_feats, att_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []
        fc_feats = fc_feats.squeeze(1)
        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats) # [50, 512]
            else:
                # mistake clone() -- solved
                it = seq[:, i-1].data.clone().contiguous()
                it = it.cpu().detach().numpy()
                it = torch.cuda.LongTensor(it)
                if i >= 2 and seq[:, i-1].sum() == 0:
                    break
                xt = self.embed(it) # [50, 512]

            output, state = self.core(xt.unsqueeze(0), state)
            output = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)
            outputs.append(output)
        # cat the outputs into one tensor
        outputs = torch.cat([_.unsqueeze(1) for _ in outputs[1:]], dim=1).contiguous()

        return outputs


