import os
import torch
import numpy as np
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs



class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(gradient_accumulation_steps=1,kwargs_handlers=[ddp_kwargs])
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # if self.accelerator.state.deepspeed_plugin is not None:
        #     self.accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"]=args.batch_size
        # print("prepare model")
        # self.model = self.accelerator.prepare(model)

    def _acquire_device(self):
        device = self.accelerator.device
        return device

    def _get_data(self, *args, **kwargs):
        pass

    def vali(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass
