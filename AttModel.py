import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from FCModel import CaptionModel
import torch.nn.init as init

class AttentiveCNN(nn.Module):
    def __init__(self, opt):
        super(AttentiveCNN, self).__init__()
        self.opt = opt
        self.avgpool = nn.AvgPool2d(7)
        self.affine_a = nn.Linear(opt.fc_feat_size, opt.embedding_size)
        self.affine_b = nn.Linear(opt.fc_feat_size, opt.rnn_size)

        self.dropout = nn.Dropout(opt.drop_prob_lm)

        self.init_weight()
    def init_weight(self):
        init.kaiming_uniform_( self.affine_a.weight, mode='fan_in' )
        init.kaiming_uniform_( self.affine_b.weight, mode='fan_in' )
        self.affine_a.bias.data.fill_( 0 )
        self.affine_b.bias.data.fill_( 0 )

    def forward(self, att_feats):
        #print(att_feats.size())
        att_size = self.opt.att_size
        att_feats = att_feats.view(-1, att_size, att_size, self.opt.fc_feat_size)
        att_feats = att_feats.transpose(1,2)
        att_feats = att_feats.transpose(1,3)
        #print(att_feats.size())
        A = att_feats

        a_g = self.avgpool(A)
        a_g = a_g.view(a_g.size(0), -1)

        # V = [ v_1, v_2, ..., v_49 ]
        V = A.view(A.size(0), A.size(1), -1).transpose(1, 2)

        # print('hidden size {}'.format(V.size()))
        # V = V.cpu()
        V = F.relu(self.affine_a(self.dropout(V)))

        v_g = F.relu(self.affine_b(self.dropout(a_g)))

        return V, v_g

class Attentive(nn.Module):
    def __init__(self, hidden_size, att_size, drop_out_lm):
        super(Attentive, self).__init__()

        self.affine_v = nn.Linear(hidden_size, att_size, bias=False)
        self.affine_g = nn.Linear(hidden_size, att_size, bias=False)
        self.affine_s = nn.Linear(hidden_size, att_size, bias=False)
        self.affine_h = nn.Linear(att_size, 1, bias=False)

        self.dropout = nn.Dropout(drop_out_lm)
        self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_normal_(self.affine_v.weight)
        torch.nn.init.xavier_normal_(self.affine_g.weight)
        torch.nn.init.xavier_normal_(self.affine_s.weight)
        torch.nn.init.xavier_normal_(self.affine_h.weight)

    def forward(self, att_feats, hiddens, sentinel):
        # affine_v [50, 196, 196]
        # hiddens [50, 17, 196]
        content_v = self.affine_v(self.dropout(att_feats)).unsqueeze(1) + self.affine_g(self.dropout(hiddens)).unsqueeze(2)
        # [50, 17, 196]
        z_t = self.affine_h(self.dropout(F.tanh(content_v))).squeeze(3)
        # TODO
        alpha_t = F.softmax(z_t.view(-1, z_t.size(2))).view(z_t.size(0), z_t.size(1), -1)
        ##alpha_t = F.softmax(z_t, dim=2)
        # alpha_t [50, 17, 196] att_feats [50, 196,512]
        # c_t [50, 17, 512]
        c_t = torch.bmm(alpha_t, att_feats).squeeze(2)

        content_s = self.affine_s(self.dropout(sentinel)) + self.affine_g(self.dropout(hiddens))
        z_t_extended = self.affine_h(self.dropout(F.tanh(content_s)))

        extended = torch.cat((z_t, z_t_extended), dim=2)
        # TODO
        alpha_hat_t = F.softmax( extended.view( -1, extended.size( 2 ) ) ).view( extended.size( 0 ), extended.size( 1 ), -1 )
        ##alpha_hat_t = F.softmax(extended, dim=2)

        beta_t = alpha_hat_t[:, :, -1]

        beta_t = beta_t.unsqueeze(2)
        c_hat_t = beta_t * sentinel + ( 1 - beta_t ) * c_t

        return c_hat_t, alpha_t, beta_t


class Sentinel(nn.Module):
    def __init__(self, input_size, hidden_size, drop_out_lm):
        super(Sentinel, self).__init__()

        self.affine_x = nn.Linear(input_size, hidden_size, bias = False)
        self.affine_h = nn.Linear(hidden_size, hidden_size, bias = False)

        self.dropout = nn.Dropout(drop_out_lm)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.affine_x.weight)
        torch.nn.init.xavier_normal_(self.affine_h.weight)

    def forward(self, x_t, h_t_1, cell_t):
        gate_t = self.affine_x(self.dropout(x_t)) + self.affine_h(self.dropout(h_t_1))
        gate_t = F.sigmoid(gate_t)

        s_t = gate_t * F.tanh(cell_t)

        return s_t


class AdaptiveBlock(nn.Module):
    def __init__(self, opt):
        super(AdaptiveBlock, self).__init__()
        self.opt = opt

        self.sentinel = Sentinel(opt.embedding_size * 2, opt.rnn_size, opt.drop_prob_lm)

        self.attentive = Attentive(opt.rnn_size, opt.att_size * opt.att_size, opt.drop_prob_lm)

        self.mlp = nn.Linear(opt.rnn_size, opt.vocab_size+1)

        self.dropout = nn.Dropout(opt.drop_prob_lm)

        self.init_weight()

    def init_weight(self):
        #torch.nn.init.xavier_normal_(self.mlp.weight)
        init.kaiming_normal_(self.mlp.weight, mode='fan_in')
        self.mlp.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.opt.rnn_type == 'lstm':
            return (weight.new_zeros(self.opt.num_layers, bsz, self.opt.rnn_size),
                    weight.new_zeros(self.opt.num_layers, bsz, self.opt.rnn_size))
        else:
            return weight.new_zeros(bsz, self.opt.rnn_size)

    def forward(self, x, hiddens, cells, att_feats, hidden_t=None):
        # transpose to make h0 batch first
        h0 = self.init_hidden(x.size(0))[0].transpose(0, 1)
        if hiddens.size( 1 ) > 1:
            hiddens_t_1 = torch.cat( ( h0, hiddens[ :, :-1, : ] ), dim=1 )
        else:
            if hidden_t is None:
                hiddens_t_1 = h0
            else:
                hiddens_t_1 = hidden_t
            
        sentinel = self.sentinel(x, hiddens_t_1, cells)

        c_hat, atten_weights, beta = self.attentive(att_feats, hiddens, sentinel)

        scores = self.mlp(self.dropout(c_hat + hiddens))

        return scores, atten_weights, beta

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.opt = opt

        self.cnn = AttentiveCNN(opt)

        self.embed = nn.Embedding(opt.vocab_size + 1, opt.embedding_size)

        self.img_embed = nn.Linear(opt.fc_feat_size, opt.embedding_size, bias = False)

        self.img_att_embed = nn.Linear(opt.fc_feat_size, opt.embedding_size, bias = False)

        self.LSTM = nn.LSTM(opt.embedding_size * 2, opt.rnn_size, 1, batch_first=True)

        self.adaptive = AdaptiveBlock(opt)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        # self.img_embed.weight.data.uniform_(-initrange, initrange)
        # self.img_att_embed.weight.data.uniform_(-initrange, initrange)

        init.kaiming_uniform_(self.img_embed.weight, mode='fan_in')
        init.kaiming_uniform_(self.img_att_embed.weight, mode='fan_in')

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.opt.rnn_type == 'lstm':
            return (weight.new_zeros(self.opt.num_layers, bsz, self.opt.rnn_size),
                    weight.new_zeros(self.opt.num_layers, bsz, self.opt.rnn_size))
        else:
            return weight.new_zeros(bsz, self.opt.rnn_size)

    def sample(self, fc_feats, att_feats, att_masks):
        #fc_embed = self.img_embed(fc_feats.squeeze(1))
        #att_feats = self.img_att_embed(att_feats)
        att_feats, fc_embed = self.cnn(att_feats)
        # x = torch.cat((embeddings, fc_embed.unsqueeze(1).expand_as(embeddings)), dim=2)
        batch_size = fc_embed.size(0)
        state = self.init_hidden(batch_size)

        seq = fc_feats.new_zeros(batch_size, self.opt.seq_length, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.opt.seq_length)

        # hiddens [batch, len, hidden] cells [len, batch, hidden]
        if torch.cuda.is_available():
            hiddens = Variable(torch.zeros( batch_size, 1, self.opt.rnn_size).cuda())
            cells = Variable(torch.zeros(1, batch_size, self.opt.rnn_size).cuda())
            it = torch.zeros(fc_feats.size(0)).long().cuda()
        else:
            hiddens = Variable(torch.zeros(batch_size, 1, self.opt.rnn_size))
            cells = Variable(torch.zeros(1, batch_size, self.opt.rnn_size))
            it = torch.zeros(fc_feats.size(0)).long()

        for time_step in range(self.opt.seq_length + 2):
            it = it.long()
            x_t = self.embed(it)
            x_t = x_t.unsqueeze(1) # [10,1,512]
            # print('time step {}'.format(time_step))

            x_t = torch.cat((x_t, fc_embed.unsqueeze(1)), dim=2)
            h_t_1 = state[0].transpose(0,1)
            h_t, state = self.LSTM(x_t, state)

            hiddens[:, 0, :] = h_t.squeeze(1)
            cells[0, :, :] = state[1]

            cells = cells.transpose(0, 1)

            if torch.cuda.device_count() > 1:
                device_ids = range(torch.cuda.device_count())
                adaptive_block_parallel = nn.DataParallel(self.adaptive, device_ids=device_ids)
                scores, atten_weights, beta = adaptive_block_parallel(x_t, hiddens, cells, att_feats, h_t_1)
            else:
                # x_t current input hiddens current hidden to control a_t cells to control sentinel att_feats V h_t_1 last hidden to use in sentinel
                scores, atten_weights, beta = self.adaptive(x_t, hiddens, cells, att_feats, h_t_1)

            cells = cells.transpose(1, 0)

            scores = scores.squeeze(1)
            Logprobs = F.log_softmax(scores, dim=1)

            if self.opt.sample_max:
                sampleLogprobs, it = torch.max(Logprobs.data, 1)
                it = it.view(-1).long()
            else:
                prob_prev = torch.exp(Logprobs.data).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = Logprobs.gather(1, it)
                it = it.view(-1).long()
            
            #iiit = it.cpu().detach().numpy()
            #print(iiit)
            
            if time_step == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            if unfinished.sum() == 0:
                break
            it = it * unfinished.type_as(it)

            seq[:,time_step] = it[:] # seq[t] the input of t+2 time step
            seqLogprobs[:,time_step] = sampleLogprobs.view(-1)
            
        return seq, seqLogprobs
  
    def _sample(self, fc_feats, att_feats, att_masks=None):
        att_feats, fc_embed = self.cnn(att_feats)
        batch_size = fc_feats.size(0)
        if torch.cuda.is_available():
            captions = Variable( torch.LongTensor( batch_size, 1 ).fill_( 0 ).cuda() )
        else:
            captions = Variable( torch.LongTensor( batch_size, 1 ).fill_( 0 ) )
        
        sampled_ids = []
        attention = []
        Beta = []
        
        # Initial hidden states
        states = None

        for i in range( 16 ):

            scores, states, atten_weights, beta = self.decoder( V, v_g, captions, states ) 
            predicted = scores.max( 2 )[ 1 ] # argmax
            captions = predicted
            
            cap = captions.cpu().detach().numpy()
            print(cap)
            
            # Save sampled word, attention map and sentinel at each timestep
            sampled_ids.append( captions )
            attention.append( atten_weights )
            Beta.append( beta )
            
        sampled_ids = torch.cat( sampled_ids, dim=1 )
        attention = torch.cat( attention, dim=1 )
        Beta = torch.cat( Beta, dim=1 )
        
        return sampled_ids
        
        
    def forward(self, mode, fc_feats, att_feats, seq, att_masks=None, state=None):
        att_feats = att_feats.float()
        if mode == 'sample':
            return self.sample(fc_feats, att_feats, att_masks)
        # [50, 17, 512]
        seq = seq.long()
        # [50, 18, 512]
        embeddings = self.embed(seq)
        # [50, 512]
        # TODO use relu to activate global feature
        ###fc_embed = F.relu(self.img_embed(fc_feats.squeeze(1)))
        ###att_feats = self.img_att_embed(att_feats)
        att_feats, fc_embed = self.cnn(att_feats)
        # [50, 17, 1024]
        x = torch.cat((embeddings, fc_embed.unsqueeze(1).expand_as(embeddings)), dim=2)

        # hiddens [batch, len, hidden] cells [len, batch, hidden]
        if torch.cuda.is_available():
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.opt.rnn_size).cuda())
            cells = Variable(torch.zeros(x.size(1), x.size(0), self.opt.rnn_size).cuda())
        else:
            hiddens = Variable(torch.zeros(x.size(0), x.size(1), self.opt.rnn_size))
            cells = Variable(torch.zeros(x.size(1), x.size(0), self.opt.rnn_size))

        for time_step in range(x.size(1)):
            x_t = x[:, time_step, :]
            x_t = x_t.unsqueeze(1)

            h_t, state = self.LSTM(x_t, state)

            hiddens[:, time_step, :] = h_t.squeeze(1)
            cells[time_step,: ,:] = state[1]
        # cells [len, batch, hidden] --> [batch, len, hidden]
        cells = cells.transpose(0, 1)

        if torch.cuda.device_count() > 1:
            device_ids = range(torch.cuda.device_count())
            adaptive_block_parallel = nn.DataParallel(self.adaptive, device_ids=device_ids)
            scores, atten_weights, beta = adaptive_block_parallel(x, hiddens, cells, att_feats)
        else:
            scores, atten_weights, beta = self.adaptive(x, hiddens, cells, att_feats)
        # return scores, state, atten_weights, beta
        scores = F.log_softmax(scores, dim=2)
        return scores[:,:-1,:]

