# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:schedular.py
@time:2020/11/22 
@desc:
'''

class DualScheduler(object):
    def __init__(self, scheduler0, scheduler1):
        self.scheduler0 = scheduler0
        self.scheduler1 = scheduler1

    def step(self):
        self.scheduler0.step()
        self.scheduler1.step()