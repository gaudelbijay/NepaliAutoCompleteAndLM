from collections import OrderedDict, namedtuple
from itertools import product
import time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class RunBuilder(object):
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


class RunManager(object):
    def __init__(self, data_size):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_start_time = None

        self.data_size = data_size

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.model = None
        self.tb = None

    def begin_run(self, run, model):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        self.model = model
        self.tb = SummaryWriter(comment=f'-{run}')

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        loss = self.epoch_loss / self.data_size
        self.tb.add_scalar('Loss ', loss, self.epoch_count)

        for name, param in self.model.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        result = OrderedDict()
        result['run'] = self.run_count
        result['epoch'] = self.epoch_count
        result['loss'] = loss
        result['epoch duration'] = epoch_duration
        result['run duration'] = run_duration

        for k, v in self.run_params._asdict().items():
            result[k] = v
        self.run_data.append(result)

    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item()*batch[0].shape[0]

    def save(self, filedir='../data/', filename='result'):
        pd.DataFrame.from_dict(self.run_data, orient='column').to_csv(f'{filedir}/{filename}.csv')
