
# Copyright (C) 2017-2023  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.


"""
Implements the co-teaching algorithm for training neural networks on noisily-labeled data (Han et al., 2018).
This module requires PyTorch (https://pytorch.org/get-started/locally/).
Example using this algorithm with cleanlab to achieve state of the art on CIFAR-10
for learning with noisy labels is provided within: https://github.com/cleanlab/examples/

``cifar_cnn.py`` provides an example model that can be trained via this algorithm.
"""

# Significant code was adapted from the following GitHub:
# https://github.com/bhanML/Co-teaching/blob/master/loss.py
# See (Han et al., 2018).

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from mil.utils_models import test_loop
import matplotlib.pyplot as plt
import numpy as np

MINIMUM_BATCH_SIZE = 16

def define_forget_rate(num_points, forget_rate = 1, start_point = None, vis = False):

    logistic_curve = lambda x, a=1.0, b=0.5: 1 / (1 + np.exp(-a * (x - b)))

    x_values = np.arange(0,num_points//4)
    logistic_values = logistic_curve(x_values, a=1.0, b=num_points//8)

    if start_point is None:
        start_point = num_points//2
    y_values = np.concatenate((np.zeros(start_point),logistic_values))
    y_values = np.concatenate((y_values, np.ones(num_points-len(y_values))))

    y_values = y_values * forget_rate

    if vis:
        plt.plot(range(0, len(y_values)), y_values)
        plt.show()

    return y_values

# Loss function for Co-Teaching
def loss_coteaching(
    y_1,
    y_2,
    t,
    forget_rate,
    class_weights=None,
):
    """Co-Teaching Loss function.

    Parameters
    ----------
    y_1 : Tensor array
      Output logits from model 1

    y_2 : Tensor array
      Output logits from model 2

    t : np.ndarray
      List of Noisy Labels (t means targets)

    forget_rate : float
      Decimal between 0 and 1 for how quickly the models forget what they learn.
      Just use rate_schedule[epoch] for this value

    class_weights : Tensor array, shape (Number of classes x 1), Default: None
      A np.torch.tensor list of length number of classes with weights
    """
    #torch.autograd.set_detect_anomaly(True)
    loss_1 = F.cross_entropy(y_1, t, reduce=False, weight=class_weights)
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted].clone()

    loss_2 = F.cross_entropy(y_2, t, reduce=False, weight=class_weights)
    ind_2_sorted = np.argsort(loss_2.data.cpu())

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # Share updates between the two models.
    # TODO: these class weights should take into account the ind_mask filters.
    loss_1_update = F.cross_entropy(y_1[ind_2_update].clone(), t[ind_2_update], weight=class_weights)
    loss_2_update = F.cross_entropy(y_2[ind_1_update].clone(), t[ind_1_update], weight=class_weights)

    return (
        torch.sum(loss_1_update) / num_remember,
        torch.sum(loss_2_update) / num_remember,
    )

    #return (torch.mean(loss_1), torch.mean(loss_2))


def initialize_lr_scheduler(lr=0.001, epochs=250, epoch_decay_start=80):
    """Scheduler to adjust learning rate and betas for Adam Optimizer"""
    mom1 = 0.9
    mom2 = 0.9  # Original author had this set to 0.1
    alpha_plan = [lr] * epochs
    beta1_plan = [mom1] * epochs
    for i in range(epoch_decay_start, epochs):
        alpha_plan[i] = float(epochs - i) / (epochs - epoch_decay_start) * lr
        beta1_plan[i] = mom2
    return alpha_plan, beta1_plan


def adjust_learning_rate(optimizer, epoch, alpha_plan, beta1_plan):
    """Scheduler to adjust learning rate and betas for Adam Optimizer"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha_plan[epoch]
        param_group["betas"] = (beta1_plan[epoch], 0.999)  # Only change beta1


def forget_rate_scheduler(epochs, forget_rate, num_gradual, exponent):
    """Tells Co-Teaching what fraction of examples to forget at each epoch."""
    # define how many things to forget at each rate schedule
    forget_rate_schedule = np.ones(epochs) * forget_rate
    forget_rate_schedule[:num_gradual] = np.linspace(0, forget_rate**exponent, num_gradual)
    return forget_rate_schedule


def calculate_accuracy(pred, labels):
    """
    Calculate accuracy given predicted values and labels.

    Args:
        pred (torch.Tensor): Predicted values (e.g., model output) of shape (batch_size, num_classes).
        labels (torch.Tensor): True labels of shape (batch_size).

    Returns:
        float: Accuracy value.
    """
    _, predicted = torch.max(pred, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

# Train the Model
def train(
    train_loader,
    epoch,
    model1,
    optimizer1,
    model2,
    optimizer2,
    args,
    class_weights = None
):
    """PyTorch training function.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
    epoch : int
    model1 : PyTorch class inheriting nn.Module
        Must define __init__ and forward(self, x,)
    optimizer1 : PyTorch torch.optim.Adam
    model2 : PyTorch class inheriting nn.Module
        Must define __init__ and forward(self, x,)
    optimizer2 : PyTorch torch.optim.Adam
    args : parser.parse_args() object
        Must contain num_iter_per_epoch, print_freq, and epochs
    forget_rate_schedule : np.ndarray of length number of epochs
        Tells Co-Teaching loss what fraction of examples to forget about.
    class_weights : Tensor array, shape (Number of classes x 1), Default: None
      A np.torch.tensor list of length number of classes with weights
    accuracy : function
        A function of the form accuracy(output, target, topk=(1,)) for
        computing top1 and top5 accuracy given output and true targets."""


    forget_rate_schedule = args.forget_rate_schedule

    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0
    running_loss_1, running_loss_2 = 0,0

    # Prepare models for training
    model1.train()
    model1.to(args.device)
    model2.train()
    model2.to(args.device)

    for i, (data, bagids, labels) in enumerate(train_loader):
        if i == len(train_loader) - 1 and len(labels) < MINIMUM_BATCH_SIZE:
            # Edge case -- the last leftover batch is small (potentially size 1)
            # This will happen if, for example, you train on 35101 examples with
            # batch size of 450. The last batch will be size 1.
            # If you update the weights based on the gradient from one example
            # if that example is noisy, you will add tons of noise to your net
            # and accuracy will actually go down with each epoch.
            # To avoid this, do not train on the last batch if it's small.
            continue

        if args.noise_func is not None:
            data = args.noise_func(data)

        data = data.to(args.device)
        labels = labels.to(args.device)
        bagids = bagids.to(args.device)

        # Forward + Backward + Optimize
        logits1 = model1((data, bagids))
        prec1= calculate_accuracy(logits1, labels)

        #prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec1
        logits2 = model2((data, bagids))
        prec2 = calculate_accuracy(logits2, labels)
        train_total2 += 1
        train_correct2 += prec2
        loss_1, loss_2 = loss_coteaching(
            logits1,
            logits2,
            labels.long().to(args.device),
            forget_rate=forget_rate_schedule[epoch],
            class_weights=class_weights,
        )

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        running_loss_1 += loss_1.item()

        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        running_loss_2 += loss_2.item()

        if (i + 1) % args.print_freq == 0:
            print('epoch: {} | loss #1: {:.3f}'.format(i + 1, running_loss_1 / len(train_loader)))
            print('epoch: {} | loss #2: {:.3f}'.format(i + 1, running_loss_2 / len(train_loader)))

    train_acc1 = float(train_correct) / float(train_total)
    train_acc2 = float(train_correct2) / float(train_total2)
    running_loss_1 = running_loss_1 / len(train_loader)
    running_loss_2 = running_loss_2 / len(train_loader)

    return (train_acc1, train_acc2), (running_loss_1, running_loss_2)


# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print("Evaluating Co-Teaching Model")
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, bagids, labels in test_loader:
        data = Variable(data).cuda()
        logits1 = model1((data, bagids))
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()  # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    for data, bagids, labels in test_loader:
        data = Variable(data).cuda()
        logits2 = model2((data, bagids))
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()

    acc1 = 100 * float(correct1) / float(total1)
    acc2 = 100 * float(correct2) / float(total2)
    return acc1, acc2