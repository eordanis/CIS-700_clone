from time import time

from models.Gan import Gan
from models.textGan_MMD.TextganDataLoader import DataLoader, DisDataloader
from models.textGan_MMD.TextganDiscriminator import Discriminator
from models.textGan_MMD.TextganGenerator import Generator
from utils.metrics.Bleu import Bleu
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.metrics.TEI import TEI
from utils.metrics.clas_acc import ACC
from utils.metrics.PPL import PPL
from utils.oracle.OracleLstm import OracleLstm
from utils.text_process import *
from utils.utils import *


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file=None, get_code=True):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)

    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer
    return codes


class TextganMmd(Gan):
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

        g_embeddings = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[self.vocab_size, self.emb_dim], stddev=0.1))
        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      g_embeddings=g_embeddings,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              g_embeddings=g_embeddings, discriminator=discriminator, start_token=self.start_token)
        self.set_generator(generator)

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
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file,
                           num_vocabulary=self.vocab_size)
        self.add_metric(docsim)
        
        tei = TEI()
        self.add_metric(tei)

        self.acc = ACC()
        self.add_metric(self.acc)
        
        ppl = PPL(self.generator_file, self.oracle_file)
        eval_samples=self.generator.sample(self.sequence_length, self.batch_size, label_i=1)
        tokens = get_tokenlized(self.generator_file)
        word_set = get_word_list(tokens)
        word_index_dict, idx2word_dict = get_dict(word_set)
        gen_tokens = tensor_to_tokens(eval_samples, idx2word_dict)
        ppl.reset(gen_tokens)
        self.add_metric(ppl)

        print("Metrics Applied: " + nll.get_name() + ", " + inll.get_name() + ", " + docsim.get_name() + ", " + tei.get_name() + ", " + self.acc.get_name() + ", " + ppl.get_name())
        
        

    def train_discriminator(self):
        for _ in range(3):
            x_batch, z_h = self.generator.generate(self.sess, True)
            y_batch = self.gen_data_loader.next_batch()
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
                self.discriminator.zh: z_h,
                self.discriminator.input_x_lable: [[1, 0] for _ in x_batch],
                self.discriminator.input_y_lable: [[0, 1] for _ in y_batch],
            }
            _ = self.sess.run(self.discriminator.train_op, feed)
            input_y,_ = self.sess.run([self.discriminator.input_label, self.discriminator.train_op], feed)
            predictions,_ = self.sess.run([self.discriminator.predictions2, self.discriminator.train_op], feed)
            self.acc.reset(predictions, input_y)

    def train_generator(self):
        z_h0 = np.random.uniform(low=-.01, high=.01, size=[self.batch_size, self.emb_dim])
        z_c0 = np.zeros(shape=[self.batch_size, self.emb_dim])

        y_batch = self.gen_data_loader.next_batch()
        feed = {
            self.generator.h_0: z_h0,
            self.generator.c_0: z_c0,
            self.generator.y: y_batch,
        }
        _ = self.sess.run(fetches=self.generator.g_updates, feed_dict=feed)
        pass

    def evaluate(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                self.log.write('epochs,')
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
                self.log.flush()
                #self.log.close()
            scores = super().evaluate()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            self.log.flush()
            return scores
        return super().evaluate()

    def train_oracle(self):
        self.init_oracle_trainng()
        self.init_metric()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.log = open(self.log_file, 'w')
        oracle_code = generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
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
                self.evaluate()

        print('Pre-training  Discriminator...')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            self.train_discriminator()

        self.reset_epoch()
        del oracle_code
        print('Adversarial Training...')
        for epoch in range(self.adversarial_epoch_num):
            start = time()
            for index in range(100):
                self.train_generator()
            end = time()
            self.add_epoch()
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

            for _ in range(15):
                self.train_discriminator()
        self.log.close()
    def init_cfg_training(self, grammar=None):
        from utils.oracle.OracleCfg import OracleCfg
        oracle = OracleCfg(sequence_length=self.sequence_length, cfg_grammar=grammar)
        self.set_oracle(oracle)
        self.oracle.generate_oracle()
        self.vocab_size = self.oracle.vocab_size + 1
        g_embeddings = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[self.vocab_size, self.emb_dim], stddev=0.1))
        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      g_embeddings=g_embeddings,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              g_embeddings=g_embeddings, discriminator=discriminator, start_token=self.start_token)
        self.set_generator(generator)

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

        self.acc = ACC()
        self.add_metric(self.acc)
        
        ppl = PPL(self.generator_file, self.test_file)
        eval_samples=self.generator.sample(self.sequence_length, self.batch_size, label_i=1)
        tokens = get_tokenlized(self.generator_file)
        word_set = get_word_list(tokens)
        word_index_dict, idx2word_dict = get_dict(word_set)
        gen_tokens = tensor_to_tokens(eval_samples, idx2word_dict)
        ppl.reset(gen_tokens)
        self.add_metric(ppl)
        
        print("Metrics Applied: " + cfg.get_name() + ", " + tei.get_name() + ", " + self.acc.get_name() + ", " + ppl.get_name())
        

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
        oracle_code = generate_samples(self.sess, self.generator, self.batch_size, self.generate_num,
                                       self.generator_file)
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

        print('Pre-training  Discriminator...')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num * 3):
            self.train_discriminator()

        self.reset_epoch()
        print('Adversarial Training...')

        del oracle_code
        for epoch in range(self.adversarial_epoch_num):
            start = time()
            for i in range(100):
                self.train_generator()
            end = time()
            self.add_epoch()
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

            for _ in range(15):
                self.train_discriminator()
        return

    def init_real_trainng(self, data_loc=None):
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        if data_loc is None:
            data_loc = 'data/image_coco.txt'
        self.sequence_length, self.vocab_size = text_precess(data_loc)

        g_embeddings = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[self.vocab_size, self.emb_dim], stddev=0.1))
        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2,
                                      emd_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      g_embeddings=g_embeddings,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              g_embeddings=g_embeddings, discriminator=discriminator, start_token=self.start_token)
        self.set_generator(generator)

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
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file,
                           num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)
        
        tei = TEI()
        self.add_metric(tei)

        self.acc = ACC()
        self.add_metric(self.acc)
        
        ppl = PPL(self.generator_file, self.oracle_file)
        eval_samples=self.generator.sample(self.sequence_length, self.batch_size, label_i=1)
        tokens = get_tokenlized(self.generator_file)
        word_set = get_word_list(tokens)
        word_index_dict, idx2word_dict = get_dict(word_set)
        gen_tokens = tensor_to_tokens(eval_samples, idx2word_dict)
        ppl.reset(gen_tokens)
        self.add_metric(ppl)
        
        print("Metrics Applied: " + inll.get_name() + ", " + docsim.get_name() + ", " + tei.get_name() + ", " + self.acc.get_name() + ", " + ppl.get_name())
        
        

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

        def get_real_code():
            text = get_tokenlized(self.oracle_file)

            def toint_list(x):
                return list(map(int, x))

            codes = list(map(toint_list, text))
            return codes

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

        print('Pre-training  Discriminator...')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            self.train_discriminator()
        oracle_code = get_real_code()


        print('Adversarial Training...')
        for epoch in range(self.adversarial_epoch_num):
            start = time()
            for index in range(100):
                self.train_generator()
            end = time()
            self.add_epoch()
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

            for _ in range(15):
                self.train_discriminator()

