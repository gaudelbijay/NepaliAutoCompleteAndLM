import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from run import RunBuilder, RunManager
import config


def train(model, criterion, epochs, X, Y):
    rm = RunManager(len(X))
    for run in RunBuilder.get_runs(config.params):
        optimizer = torch.optim.Adam(model.parameters(), lr=run.lr)
        train_X, train_Y = utils.batchify(X, Y)

        rm.begin_run(run, model)
        for e in range(epochs):
            rm.begin_epoch()
            for i in range(len(train_X)):
                x = train_X[i]
                y = train_Y[i]
                optimizer.zero_grad()
                prediction, hidden = model(x)
                loss = criterion(prediction, y)
                loss.backward()
                optimizer.step()
                rm.track_loss(loss, x)
            rm.end_epoch()
        rm.end_run()
    rm.save()
