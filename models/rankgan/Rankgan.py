from time import time

from models.Gan import Gan
from models.rankgan.RankganDataLoader import DataLoader, DisDataloader
from models.rankgan.RankganDiscriminator import Discriminator
from models.rankgan.RankganGenerator import Generator
from models.rankgan.RankganReward import Reward
from utils.metrics.Bleu import Bleu
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.metrics.TEI import TEI
from utils.metrics.PPL import PPL
from utils.oracle.OracleLstm import OracleLstm
from utils.text_process import *
from utils.utils import *


class Rankgan(Gan):
    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 32
        self.hidden_dim = 32
        self.sequence_length = 20
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 64
        self.generate_num = 128
        self.start_token = 0


    def init_oracle_trainng(self, oracle=None):
        if oracle is None:
            oracle = OracleLstm(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                                hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                start_token=self.start_token)
        self.set_oracle(oracle)

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters, batch_size=self.batch_size,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

    def init_metric(self):

        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)
        
        tei = TEI()
        self.add_metric(tei)
        
        ppl = PPL(self.generator_file, self.oracle_file)
        # eval_samples = [self.generator.sample(self.sequence_length * self.batch_size, self.batch_size, label_i=i)
        #                 for i in range(2)]
        eval_samples=self.generator.sample(self.sequence_length, self.batch_size, label_i=1)
        tokens = get_tokenlized(self.generator_file)
        word_set = get_word_list(tokens)
        word_index_dict, idx2word_dict = get_dict(word_set)
        gen_tokens = tensor_to_tokens(eval_samples, idx2word_dict)
        ppl.reset(gen_tokens)
        self.add_metric(ppl)

        print("Metrics Applied: " + nll.get_name() + ", " + inll.get_name() + ", " + docsim.get_name() + ", " + tei.get_name() + ", " + ppl.get_name())
        
        

    def train_discriminator(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
        for _ in range(3):
            self.dis_data_loader.next_batch()
            x_batch, y_batch, ref_batch = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
                self.discriminator.input_ref: ref_batch,
            }
            _ = self.sess.run(self.discriminator.train_op, feed)

    def evaluate(self):
        #print("evaluate")
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            #print("write to file")
            if self.epoch == 0 or self.epoch == 1:
                self.log.write('epochs,')
                for metric in self.metrics:
                    #print(metric.get_name())
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
                self.log.flush()
                #self.log.close()
            scores = super().evaluate()
            for score in scores:
                #print("score")
                self.log.write(str(score)+',')
            self.log.write('\n')
            self.log.flush()
            #self.log.close()
            return scores
        #self.log.close()
        #print("end evaluate")
        return super().evaluate()

    def train_oracle(self):
        self.init_oracle_trainng()
        self.init_metric()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.log = open(self.log_file, 'w')
        generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)

        rollout = Reward(self.generator, .8)
        print('Pre-training  Generator...')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            self.add_epoch()
            if epoch % 5 == 0:
                self.evaluate()

        print('Pre-training   Discriminator...')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            self.train_discriminator()


        print('Adversarial Training...')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator, self.dis_data_loader)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                _ = self.sess.run(self.generator.g_updates, feed_dict=feed)
            end = time()
            self.add_epoch()
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num-1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()
        self.log.close()
    def init_cfg_training(self, grammar=None):
        from utils.oracle.OracleCfg import OracleCfg
        oracle = OracleCfg(sequence_length=self.sequence_length, cfg_grammar=grammar)
        self.set_oracle(oracle)
        self.oracle.generate_oracle()
        self.vocab_size = self.oracle.vocab_size + 1

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        return oracle.wi_dict, oracle.iw_dict

    def init_cfg_metric(self, grammar=None):
        from utils.metrics.Cfg import Cfg
        cfg = Cfg(test_file=self.test_file, cfg_grammar=grammar)
        self.add_metric(cfg)
        
        tei = TEI()
        self.add_metric(tei)
        
        ppl = PPL(self.generator_file, self.test_file)
        # eval_samples = [self.generator.sample(self.sequence_length * self.batch_size, self.batch_size, label_i=i)
        #                 for i in range(2)]
        eval_samples=self.generator.sample(self.sequence_length, self.batch_size, label_i=1)
        tokens = get_tokenlized(self.generator_file)
        word_set = get_word_list(tokens)
        word_index_dict, idx2word_dict = get_dict(word_set)
        gen_tokens = tensor_to_tokens(eval_samples, idx2word_dict)
        ppl.reset(gen_tokens)
        self.add_metric(ppl)
        
        print("Metrics Applied: " + cfg.get_name() + ", " + tei.get_name() + ", " + ppl.get_name())
        

    def train_cfg(self):
        import json
        from utils.text_process import get_tokenlized
        from utils.text_process import code_to_text
        cfg_grammar = """
          S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x | '(' S ')'
          PLUS -> '+'
          SUB -> '-'
          PROD -> '*'
          DIV -> '/'
          x -> 'x' | 'y'
        """

        wi_dict_loc, iw_dict_loc = self.init_cfg_training(cfg_grammar)
        with open(iw_dict_loc, 'r') as file:
            iw_dict = json.load(file)

        def get_cfg_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        self.init_cfg_metric(grammar=cfg_grammar)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.log = open(self.log_file, 'w')
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)
        print('Pre-training  Generator...')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

        print('Pre-training   Discriminator...')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num * 3):
            self.train_discriminator()

        self.reset_epoch()
        print('Adversarial Training...')
        self.reward = Reward(self.generator, .8)
        for epoch in range(self.adversarial_epoch_num):
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator, self.dis_data_loader)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                _ = self.sess.run(self.generator.g_updates, feed_dict=feed)
            end = time()
            self.add_epoch()
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()
        return

    def init_real_trainng(self, data_loc=None):
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        from utils.text_process import get_dict
        if data_loc is None:
            data_loc = 'data/image_coco.txt'
        self.sequence_length, self.vocab_size = text_precess(data_loc)
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters, batch_size=self.batch_size,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = None
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        tokens = get_tokenlized(data_loc)
        word_set = get_word_list(tokens)
        [word_index_dict, index_word_dict] = get_dict(word_set)
        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, word_index_dict, self.sequence_length))
        return word_index_dict, index_word_dict

    def init_real_metric(self):
        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)
        
        tei = TEI()
        self.add_metric(tei)
        
        ppl = PPL(self.generator_file, self.oracle_file)
        # eval_samples = [self.generator.sample(self.sequence_length * self.batch_size, self.batch_size, label_i=i)
        #                 for i in range(2)]
        eval_samples=self.generator.sample(self.sequence_length, self.batch_size, label_i=1)
        tokens = get_tokenlized(self.generator_file)
        word_set = get_word_list(tokens)
        word_index_dict, idx2word_dict = get_dict(word_set)
        gen_tokens = tensor_to_tokens(eval_samples, idx2word_dict)
        ppl.reset(gen_tokens)
        self.add_metric(ppl)
        
        print("Metrics Applied: " + inll.get_name() + ", " + docsim.get_name() + ", " + tei.get_name() + ", " + ppl.get_name())
        
        

    def train_real(self, data_loc=None):
        from utils.text_process import code_to_text
        from utils.text_process import get_tokenlized
        wi_dict, iw_dict = self.init_real_trainng(data_loc)
        self.init_real_metric()

        def get_real_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.log = open(self.log_file, 'w')
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)

        print('Pre-training  Generator...')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_real_test_file()
                self.evaluate()

        print('Pre-training   Discriminator...')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            self.train_discriminator()

        self.reset_epoch()
        print('Adversarial Training...')
        try:
         self.reward = Reward(self.generator, .8)
         for epoch in range(self.adversarial_epoch_num):
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator, self.dis_data_loader)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                _ = self.sess.run(self.generator.g_updates, feed_dict=feed)
            end = time()
            self.add_epoch()
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_real_test_file()
                self.evaluate()

            self.reward.update_params()
            for _ in range(15):
                self.train_discriminator()
        except Exception as e:
            print("evaluation error")
            print(e)
