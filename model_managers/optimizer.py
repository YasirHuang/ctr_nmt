# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:optimizer.py
@time:2020-09-15 
@desc:
'''


class NoamOpt:
    def __init__(self, d_model, factor, warmup, optimizer, grad_clip=-1.0, delay_update=1):
        self.d_model = d_model
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self.grad_clip = grad_clip

        # update steps
        self._step = 0
        self._rate = 0

        # for delay update
        self._step_delay = 0
        self.delay_update = delay_update

    def step(self):
        '''
         update learning rate first, then apply optimizer
        :return:
        '''

        self._step_delay += 1

        if self._step_delay % self.delay_update == 0:
            self._step += 1
            lrate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = lrate
            self._rate = lrate
            #
            # if self.grad_clip > 0:
            #     for p in self.optimizer.param_groups:
            #         nn.utils.clip_grad_norm_(p['params'], self.grad_clip)

            self.optimizer.step()
            self.optimizer.zero_grad()

    def rate(self, step=None):
        '''
            lr = xxx
        '''
        if step is None:
            step = self._step

        lr = self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))

        return self.factor * lr

    def set_steps(self, batch_steps):
        '''
        when resume training, need this
        :param step:
        :return:
        '''
        self._step = batch_steps // self.delay_update
        self._step_delay = batch_steps

    def decay(self, r):
        self.factor *= r

    def zero_grad(self):
        return

    def state_dict(self):
        optim_stats = self.optimizer.state_dict()
        return optim_stats

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

class DualNoamOpt:
    def __init__(self, transformer_noamopt:NoamOpt, mmtransformer_noamopt:NoamOpt):
        self.transformer_noamopt = transformer_noamopt
        self.mmtransformer_noamopt = mmtransformer_noamopt

        self.current_optimizer = self.transformer_noamopt

    def set_current_optimizer(self, current_optimizer):
        self.current_optimizer = current_optimizer

    def step(self):
        self.current_optimizer.step()
        # if self.current_mode == titm.MODEL_TRAINING_MODE.NMT:
        #     self.transformer_noamopt.step()
        # elif self.current_mode == titm.MODEL_TRAINING_MODE.IMAGINATE:
        #     self.mmtransformer_noamopt.step()
        # else:
        #     raise ValueError("Unknown current mode %s" % self.current_mode)

    def rate(self, step=None):
        return self.current_optimizer.rate(step)
        # if self.current_mode == titm.MODEL_TRAINING_MODE.NMT:
        #     return self.transformer_noamopt.rate(step)
        # elif self.current_mode == titm.MODEL_TRAINING_MODE.IMAGINATE:
        #     return self.mmtransformer_noamopt.rate(step)
        # else:
        #     raise ValueError("Unknown current mode %s" % self.current_mode)

    def set_steps(self, batch_steps):
        self.current_optimizer.set_steps(batch_steps)
        # if self.current_mode == titm.MODEL_TRAINING_MODE.NMT:
        #     self.transformer_noamopt.set_steps(batch_steps)
        # elif self.current_mode == titm.MODEL_TRAINING_MODE.IMAGINATE:
        #     self.mmtransformer_noamopt.set_steps(batch_steps)
        # else:
        #     raise ValueError("Unknown current mode %s" % self.current_mode)

    def decay(self, r):
        self.current_optimizer.decay(r)
        # if self.current_mode == titm.MODEL_TRAINING_MODE.NMT:
        #     self.transformer_noamopt.decay(r)
        # elif self.current_mode == titm.MODEL_TRAINING_MODE.IMAGINATE:
        #     self.mmtransformer_noamopt.decay(r)
        # else:
        #     raise ValueError("Unknown current mode %s" % self.current_mode)

    def zero_grad(self):
        return self.current_optimizer.zero_grad()
        # if self.current_mode == titm.MODEL_TRAINING_MODE.NMT:
        #     return self.transformer_noamopt.zero_grad()
        # elif self.current_mode == titm.MODEL_TRAINING_MODE.IMAGINATE:
        #     return self.mmtransformer_noamopt.zero_grad()
        # else:
        #     raise ValueError("Unknown current mode %s" % self.current_mode)

class DualOptimizer(object):

    def __init__(self, singlemodal_optimizer, multimodal_optimizer):
        self.singlemodal_optimizer = singlemodal_optimizer
        self.multimodal_optimizer = multimodal_optimizer

        self.current_optimizer = self.singlemodal_optimizer

    def set_current_optimizer(self, current_optimizer):
        assert current_optimizer is self.singlemodal_optimizer or \
               current_optimizer is self.multimodal_optimizer
        self.current_optimizer = current_optimizer

    def step(self):
        self.current_optimizer.step()

    def rate(self, step=None):
        return self.current_optimizer.rate(step)

    def set_steps(self, batch_steps):
        self.current_optimizer.set_steps(batch_steps)

    def decay(self, r):
        self.current_optimizer.decay(r)

    def zero_grad(self):
        return self.current_optimizer.zero_grad()
