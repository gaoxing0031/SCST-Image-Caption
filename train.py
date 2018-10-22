import time
import torch.optim as optim
from dataloader import *
import tensorboardX as tb
from opts import Opt
from FCModel import *
from AttModel import *
import utils
import eval
import torch.nn as nn
from rewards import *
import sys

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    loader = DataLoader(opt)
    tb_summary_writer = tb.SummaryWriter(opt.checkpoint_path)

    infos ={}
    histories = {}

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)

    # model = FCModel(opt).cuda()
    model =  AttModel(opt).cuda()
    #dp_model = torch.nn.DataParallel(model)
    dp_model = model
    dp_model.train()

    crit = utils.LanguageModelCriterion()
    rl_crit = RewardCriterion()
    optimizer = optim.Adam(model.parameters(), opt.learning_rate, (0.9, 0.999), 1e-8, weight_decay=0)

    sc_flag = False

    start = time.time()
    while True:
        # sys.stdout.flush()
        # Learning rate decay
        if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            decay_factor = opt.learning_rate_decay_rate ** frac
            opt.current_lr = opt.learning_rate * decay_factor
        else:
            opt.current_lr = opt.learning_rate

        # Start use SCST to train
        if opt.self_critical_after >= 0 and epoch >= opt.self_critical_after:
            sc_flag = True
            init_scorer(opt.cached_tokens)
        else:
            sc_flag = False
        ##
        # sc_flag = True
        # init_scorer(opt.cached_tokens)
        ##
        utils.set_lr(optimizer, opt.current_lr)

        data = loader.get_batch('train')

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp

        optimizer.zero_grad()
        if not sc_flag:
            loss = crit(dp_model('forward', fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:])
            # loss = crit(dp_model('forward', fc_feats, att_feats, labels, att_masks), labels, masks)
            # loss = crit(dp_model(fc_feats, att_feats, labels, att_masks), labels[:, 1:], masks[:, 1:])
        else:
            # Generate baseline with argmax
            opt.sample_max = False
            gen_result, sample_logprobs = dp_model('sample', fc_feats, att_feats, labels, att_masks)
            opt.sample_max = True
            reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(dp_model.parameters(), opt.grad_clip)

        train_loss = loss.item()

        optimizer.step()

        if iteration % opt.print_every == 0:
            torch.cuda.synchronize()
            end = time.time()
            if not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, train_loss, end - start))
            else:
                print("iter {} (epoch {}), train_loss = {:.3f}, avg_reward = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, train_loss, np.mean(reward[:, 0]), end - start))
            start = time.time()

        iteration += 1

        if data['bounds']['wrapped']:
            epoch += 1
        #-------------------------------------------------------------------#
        if (iteration % opt.checkpoint_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)

            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr

        #-------------------------------------------------------------------#
        if (iteration % opt.save_every == 0):
            val_loss, predictions, lang_stats = eval.eval_split(dp_model, crit, loader, 'val', opt)
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)


            if lang_stats is not None:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
                    print('{} : {}'.format(k, v))
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            current_score = lang_stats['CIDEr']
            if current_score > opt.best_cider_score:
                print('New Best Cider Score: {}'.format(current_score))
                opt.best_cider_score = current_score
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print('Save best model!')

        if epoch >= opt.max_epochs and opt.max_epochs >= 0:
            break

if __name__ == '__main__':
    opt = Opt()
    train(opt)
