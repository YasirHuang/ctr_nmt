# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:basic_training_manager.py
@time:2020-08-29 
@desc:
'''
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from utils.config import MODE, DATA_PART
from utils import utils, config, early_stop
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from model_managers.optimizer import NoamOpt
from model_managers.utils import ppl as calculate_ppl

from models.loss_functions import LabelSmoothing

from procedures.eval_procedure import eval_procedure
from procedures.infer_procedure import infer_procedure


class BasicTrainingManager:
    def __init__(self, mode: MODE, FLAGS, log):
        self.mode = mode
        self.log = log

        self.out_dir = FLAGS.out_dir
        self.criterion = FLAGS.criterion
        self.smoothing = FLAGS.smoothing
        self.validate_with_evaluation = FLAGS.validate_with_evaluation
        self.validate_with_inference = FLAGS.validate_with_inference
        self.stop_signal = FLAGS.stop_signal

        # training parameters
        self.start_decay_step = FLAGS.start_decay_step
        self.optimize_delay = FLAGS.optimize_delay
        self.scheduler_type = FLAGS.scheduler_type
        self.loss_normalize_type = FLAGS.loss_normalize_type
        self.max_gradient_norm = FLAGS.max_gradient_norm
        self.adam_beta1 = FLAGS.adam_beta1
        self.adam_beta2 = FLAGS.adam_beta2

        self.tgt_vocab_size = FLAGS.tgt_vocab_size
        self.pad_token_id = FLAGS.pad_token_id
        self.multiple_gpu = FLAGS.multiple_gpu

        self.summary_writer = self.create_summary_writer(self.out_dir)
        self.device = self.create_device()

        self.model = None

        self.criterions = None
        self.optimizer = None

        self.training_running_loss = 0.0
        self.evaluating_running_loss = 0.0

        self.training_num_of_tokens = 0
        self.evaluating_num_of_tokens = 0

        self.training_num_of_sents = 0
        self.evaluating_num_of_sents = 0

        self.training_total_step = 0
        self.evaluating_total_step = 0
        self.testing_total_step = 0

        self.steps_per_internal_eval = 0
        self.stop_flag = False
        self.stoper = None
        self.scheduler = None
        self.training_start_time = None
        self.evaluating_start_time = None
        self.testing_start_time = None
        self.validation_function_chain = {}

    def training_one_more_step(self, batch_info):
        loss = batch_info['loss'].data.clone()
        num_of_tokens = batch_info['num_of_tokens'] if 'num_of_tokens' in batch_info else 1
        num_of_sents = batch_info['num_of_sents'] if 'num_of_sents' in batch_info else 1
        self.training_running_loss += self.denormalize_loss(loss, num_of_sents, num_of_tokens)
        self.training_num_of_sents += num_of_sents
        self.training_num_of_tokens += num_of_tokens
        self.training_total_step += 1
        self.steps_per_internal_eval += 1

    def evaluating_one_more_step(self, batch_info):
        loss = batch_info['loss'].data.clone()
        num_of_tokens = batch_info['num_of_tokens'] if 'num_of_tokens' in batch_info else 1
        num_of_sents = batch_info['num_of_sents'] if 'num_of_sents' in batch_info else 1
        self.evaluating_running_loss += self.denormalize_loss(loss, num_of_sents, num_of_tokens)
        self.evaluating_num_of_sents += num_of_sents
        self.evaluating_num_of_tokens += num_of_tokens
        self.evaluating_total_step += 1

    def testing_one_more_step(self):
        self.testing_total_step += 1

    def reset_internal_evalation(self):
        self.training_running_loss = 0.0
        self.training_num_of_tokens = 0
        self.training_num_of_sents = 0
        self.steps_per_internal_eval = 0

    def reset_external_evalation(self):
        self.evaluating_running_loss = 0.0
        self.evaluating_num_of_tokens = 0
        self.evaluating_num_of_sents = 0
        self.evaluating_total_step = 0

    def create_summary_writer(self, logdir):
        writer = SummaryWriter(logdir)
        return writer

    def create_device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_criterions(self, mode=None):
        if mode is None:
            mode = self.mode
        # create the criterion for training/evaluating mode
        if mode == MODE.Train or mode == MODE.Eval:
            self.criterions = {}
            if self.criterion == 'CrossEntropyLoss':
                self.criterions['CrossEntropyLoss'] = nn.CrossEntropyLoss()
            elif self.criterion == 'NLLLoss':
                # weight = torch.ones(self.tgt_vocab_size)
                # weight[self.pad_token_id] = 0
                self.criterions['NLLLoss'] = nn.NLLLoss(size_average=False)
            elif self.criterion == 'LabelSmoothing':
                self.criterions['LabelSmoothing'] = LabelSmoothing(smoothing=self.smoothing)
            elif self.criterion == 'MSELoss':
                self.criterions['MSELoss'] = nn.MSELoss()
        else:
            self.criterions = None
        return self.criterions

    def create_optimizer(self, optimizer_name, learning_rate, model):
        if optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                       lr=learning_rate, momentum=0.9)
        elif optimizer_name.lower() == 'adadelta':
            self.optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                            lr=learning_rate)
        elif optimizer_name.lower() == 'adagrad':
            self.optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=learning_rate)
        elif optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                                        betas=(self.adam_beta1, self.adam_beta2), eps=1e-9)
        elif optimizer_name.lower() == 'noamadam':
            adamopt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 betas=(self.adam_beta1, self.adam_beta2), eps=1e-9)
            d_model = getattr(self, 'd_model')
            self.optimizer = NoamOpt(d_model,
                                     learning_rate,
                                     self.start_decay_step,
                                     adamopt,
                                     delay_update=self.optimize_delay)
        else:
            raise ValueError('Optimizer unsupported: %s' % optimizer_name)
        return self.optimizer

    def get_learning_rate(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def report_internal_evaluation(self, model, current_epoch, steps_in_epoch, summary_writer=None):
        if summary_writer is None:
            summary_writer = self.summary_writer

        # normed_loss = self.normalize_loss(self.training_running_loss,
        #                                   self.training_num_of_sents,
        #                                   self.training_num_of_tokens)
        tls = self.training_running_loss / self.training_num_of_tokens
        sls = self.training_running_loss / self.training_num_of_sents
        ppl = calculate_ppl(self.training_running_loss, self.training_num_of_tokens)
        lr = self.get_learning_rate()
        print("training info: epoch %d (%d/%d): sloss %.2f, tloss %.2f, ppl %.2f, lr %.1e, time/step %.2fs" %
              (current_epoch,
               steps_in_epoch,
               self.training_total_step,
               sls, tls,
               ppl,
               lr,
               (time.time() - self.training_start_time) / self.steps_per_internal_eval))
        self.reset_internal_evalation()
        self.now_as_start_time(MODE.Train)
        summary_writer.add_scalar(tag="train_token_loss", scalar_value=tls, global_step=self.training_total_step)
        summary_writer.add_scalar(tag="train_sentence_loss", scalar_value=sls, global_step=self.training_total_step)
        summary_writer.add_scalar(tag="train_ppl", scalar_value=ppl, global_step=self.training_total_step)

        sys.stdout.flush()
        if self.loss_normalize_type == 'token':
            return tls
        else:
            return sls

    def report_external_evaluation(self, model, summary_writer=None, global_step=None):
        if summary_writer is None:
            summary_writer = self.summary_writer
        if global_step is None:
            global_step = self.training_total_step

        # normed_loss = self.normalize_loss(self.evaluating_running_loss,
        #                                   self.evaluating_num_of_sents,
        #                                   self.evaluating_num_of_tokens)
        # loss = normed_loss
        tls = self.evaluating_running_loss / self.evaluating_num_of_tokens
        sls = self.evaluating_running_loss / self.evaluating_num_of_sents
        ppl = calculate_ppl(self.evaluating_running_loss, self.evaluating_num_of_tokens)
        print("evaluation info: tloss %.2f, sloss %.2f, ppl %.2f, eval step %d total time %.2fs (%d steps)" %
              (tls, sls,
               ppl,
               self.evaluating_total_step,
               (time.time() - self.evaluating_start_time) / self.evaluating_total_step,
               self.evaluating_total_step))
        self.reset_external_evalation()

        summary_writer.add_scalar(tag="eval_token_loss", scalar_value=tls, global_step=global_step)
        summary_writer.add_scalar(tag="eval_sentence_loss", scalar_value=sls, global_step=global_step)
        summary_writer.add_scalar(tag="eval_ppl", scalar_value=ppl, global_step=global_step)

        sys.stdout.flush()
        if self.loss_normalize_type == 'token':
            return tls
        else:
            return sls

    def report_inference(self,
                         model,
                         inference_results,
                         name="inference bleu_score",
                         summary_writer=None,
                         global_step=None):
        if summary_writer is None:
            summary_writer = self.summary_writer
        if global_step is None:
            global_step = self.training_total_step
        utils.print_out('%s result (time %.2fs): %.2f' % (name,
                                                          time.time() - self.testing_start_time,
                                                          inference_results))
        summary_writer.add_scalar(tag=name,
                                  scalar_value=inference_results,
                                  global_step=global_step)

    def now_as_start_time(self, mode):
        if mode == MODE.Train:
            self.training_start_time = time.time()
        elif mode == MODE.Eval:
            self.evaluating_start_time = time.time()
        elif mode == MODE.Infer:
            self.testing_start_time = time.time()
        else:
            raise ValueError("Unknown mode %s" % str(mode))

    def set_stoper(self, max_steps_without_change):
        if self.stop_signal == 'bleu':
            self.stoper = early_stop.EarlyStopController(
                max_steps_without_change,
                stop_type=early_stop.STOP_TYPE.INCREASE)
        elif self.stop_signal == 'loss':
            self.stoper = early_stop.EarlyStopController(
                max_steps_without_change,
                stop_type=early_stop.STOP_TYPE.DECREASE)
        elif self.stop_signal == 'step':
            self.stoper = early_stop.StepStopController(
                max_steps_without_change * self.optimize_delay)
        return self.stoper

    def stop_training(self):
        self.stop_flag = True

    def is_training_stopped(self):
        return self.stop_flag

    def set_scheduler(self, optimizer, step_size, gamma):
        if self.scheduler_type is None or not isinstance(optimizer, optim.Optimizer):
            return None
        elif self.scheduler_type == "StepLR":
            self.scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif self.scheduler_type == "ExponentialLR":
            self.scheduler = ExponentialLR(optimizer, gamma=gamma)
        else:
            raise ValueError("Unsupported scheduler type %s." % self.scheduler_type)
        return self.scheduler

    def parallel_model(self, model=None):
        if model is not None:
            assert model is self.model, "not the same object model."
        if self.multiple_gpu:
            model = nn.DataParallel(model)
            self.model = model
        return model

    def freeze_parameters(self, model=None):
        pass

    def load_pretrained_model(self, model, dataset_manager):
        pass

    def to_device(self, model, criterions=None, device=None):
        if device is None:
            device = self.device
        # model = model.to(device)
        model.to(device)
        if criterions is not None:
            criterions[self.criterion].to(device)
        #     criterions['CrossEntropyLoss'] = criterions['CrossEntropyLoss'].to(device)
        #     return model, criterions
        # return model

    def validate(self,
                 FLAGS,
                 dataset_part=DATA_PART.Val,
                 training_manager=None,
                 dataset_manager=None,
                 checkpoint_name=None,
                 global_step=0):
        if self.validate_with_evaluation:
            eval_loss = eval_procedure(FLAGS,
                                       dataset_part=dataset_part,
                                       training_manager=training_manager,
                                       dataset_manager=dataset_manager,
                                       checkpoint_name=checkpoint_name,
                                       global_step=global_step)
        else:
            eval_loss = 0
        if self.validate_with_inference:
            bleu_score = infer_procedure(FLAGS,
                                         dataset_part=dataset_part,
                                         training_manager=training_manager,
                                         dataset_manager=dataset_manager,
                                         checkpoint_name=checkpoint_name,
                                         global_step=global_step)
        else:
            bleu_score = 0
        if self.stop_signal == "loss":
            assert self.validate_with_evaluation
            assert self.stoper.stop_type == early_stop.STOP_TYPE.DECREASE
            return eval_loss
        elif self.stop_signal == "bleu":
            assert self.validate_with_inference
            assert self.stoper.stop_type == early_stop.STOP_TYPE.INCREASE
            return bleu_score
        elif self.stop_signal == "step":
            return self.training_total_step
        else:
            raise ValueError("Unknown stop signal %s" % self.stop_signal)

    def test(self,
             FLAGS,
             dataset_part=DATA_PART.Test,
             training_manager=None,
             dataset_manager=None,
             checkpoint_name=None,
             global_step=0):
        infer_procedure(FLAGS,
                        dataset_part=dataset_part,
                        training_manager=training_manager,
                        dataset_manager=dataset_manager,
                        checkpoint_name=checkpoint_name,
                        global_step=global_step)

    def normalize_loss(self, loss, number_of_sentences, number_of_tokens):
        if self.loss_normalize_type == 'token':
            return loss / number_of_tokens
        elif self.loss_normalize_type == 'sentence':
            return loss / number_of_sentences
        else:
            return loss

    def denormalize_loss(self, loss, number_of_sentences, number_of_tokens):
        if self.loss_normalize_type == 'token':
            return loss * number_of_tokens
        elif self.loss_normalize_type == 'sentence':
            return loss * number_of_sentences
        else:
            return loss

    def backward(self, batch_info, model=None):
        model = self.model if model is None else model
        loss = batch_info['loss']
        loss.backward()
        if self.max_gradient_norm > 0:
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                           max_norm=self.max_gradient_norm)

    def loss_forward(self, *args, **kwargs):
        return NotImplementedError

    def hypo_forward(self, *args, **kwargs):
        return NotImplementedError
