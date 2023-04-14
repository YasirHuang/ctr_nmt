# coding=utf-8
'''
@author:Xin Huang
@contact:xin.huang@nlpr.ia.ac.cn
@file:early_stop.py
@time:2019/2/18 
@desc:
'''
from utils import utils
from enum import Enum


# MODE = Enum('MODE', ('Train', 'Val', 'Test'))
class STOP_TYPE(Enum):
    DECREASE = 0
    INCREASE = 1


class EarlyStopController:

    def __init__(self,
                 max_steps_without_change,
                 min_steps=None,
                 stop_type: STOP_TYPE = STOP_TYPE.DECREASE):
        self.max_steps_without_change = max_steps_without_change
        self.min_steps = min_steps if min_steps is not None else 2
        self.metrics = []
        self.the_best = None
        self.continue_counter = 0
        self.step = 0
        self.is_stoped = False
        self.stop_type = stop_type

    def if_decrease_stoped(self, metric=None):
        '''
        If the metric has been stopped decreasing for max_steps_without_change steps,
        then return True, else return False.
        :param metric:
        :return:
        '''
        if metric is None:
            return self.is_stoped
        self.step += 1
        self.metrics.append(metric)
        if self.the_best is None:
            self.the_best = metric
            return False
        if metric < self.the_best:
            self.the_best = metric
            self.continue_counter = 0
            return False
        else:
            self.continue_counter += 1
            utils.print_out("current continue counter is %d, the best metric value is %.2f" % (
            self.continue_counter, self.the_best))
            if self.continue_counter >= self.max_steps_without_change:
                utils.print_out('stop training while the step is %d, which without decrease for %d steps'
                                % (self.step, self.continue_counter))
                self.is_stoped = True
                return True
            return False

    def if_increase_stoped(self, metric=None):
        '''
        If the metric has been stopped increasing for max_steps_without_change steps,
        then return True, else return False.
        :param metric:
        :return:
        '''
        if metric is None:
            return self.is_stoped
        self.step += 1
        self.metrics.append(metric)
        if self.the_best is None:
            self.the_best = metric
            return False
        if metric > self.the_best:
            self.the_best = metric
            self.continue_counter = 0
            return False
        else:
            self.continue_counter += 1
            utils.print_out("current continue counter is %d, the best metric value is %.2f" % (
            self.continue_counter, self.the_best))
            if self.continue_counter >= self.max_steps_without_change:
                utils.print_out('stop training while the step is %d, which without increase for %d steps'
                                % (self.step, self.continue_counter))
                self.is_stoped = True
                return True
            return False

    def if_stoped(self, metric=None):
        if self.stop_type == STOP_TYPE.DECREASE:
            return self.if_decrease_stoped(metric)
        elif self.stop_type == STOP_TYPE.INCREASE:
            return self.if_increase_stoped(metric)
        else:
            raise ValueError("Unknown stop_type %s" % self.stop_type)

class StepStopController:
    def __init__(self,max_step):
        self.max_step = max_step
        self.is_stoped = False

    def if_stoped(self, step=None):
        if step is None:
            return self.is_stoped
        if step < self.max_step:
            self.is_stoped = False
        else:
            self.is_stoped = True
        return self.is_stoped
