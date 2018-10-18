
class Opt():
    def __init__(self):
        self.vocab_size = 0
        self.seq_length = 0
        self.batch_size = 10 # 50 to train in one batch
        self.seq_per_img = 5

        self.input_json_file = 'G:/Python/critical/data/cocotalk.json'
        self.input_label_file = 'G:/Python/critical/data/cocotalk_label.h5'
        self.input_fc_dir = 'G:/Python/coco_resnet101/data_fc'
        self.input_att_dir = 'G:/Python/coco_resnet101/data_att'
        self.checkpoint_path = './log'
        

        self.use_att = False
        self.norm_att_feat = False

        self.rnn_type='lstm'
        self.embedding_size = 512
        self.input_encoding_size = 512 ###
        self.fc_feat_size = 2048
        self.att_size = 14
        self.num_layers = 1
        self.rnn_size = 512
        self.drop_prob_lm = 0.5
        self.ss_prob = 0.0 # Schedule sampling probability

        self.max_epochs = 20
        self.print_every = 50
        self.print_eval_every = 1000
        self.checkpoint_every = 100
        self.save_every = 100
        self.max_eval_points = 0
        self.grad_clip = 0.5
        self.learning_rate = 5e-5
        self.current_lr = 0
        self.learning_rate_decay_start = 10
        self.learning_rate_decay_every = 5
        self.learning_rate_decay_rate = 0.8
        # Greedy Search
        self.sample_max = True

        self.beam_size = 1

        self.use_att = False
        self.norm_att_feat = False

        self.val_num_images = 5000
        self.self_critical_after=10
        self.cached_tokens = 'coco-all-idxs'
        self.cider_reward_weight = 1
        self.bleu_reward_weight = 0
        self.best_cider_score = 0