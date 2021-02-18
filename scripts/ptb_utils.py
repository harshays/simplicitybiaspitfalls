import seaborn as sns
import utils
import random
import os, copy, pickle, time
import itertools
from collections import defaultdict, Counter, OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torch import optim, nn
import torch.nn.functional as F

import gpu_utils as gu
import data_utils as du
import synth_models

#import foolbox as fb
#from autoattack import AutoAttack

# Misc
def get_yhat(model, data): return torch.argmax(model(data), 1)
def get_acc(y,yhat): return (y==yhat).sum().item()/float(len(y))

class PGD_Attack(object):

    def __init__(self, eps, lr, num_iter, loss_type, rand_eps=1e-3,
                 num_classes=2, bounds=(0.,1.), minimal=False, restarts=1, device=None):
        self.eps = eps
        self.lr = lr
        self.num_iter = num_iter
        self.B = bounds
        self.restarts = restarts
        self.rand_eps = rand_eps
        self.device = device or gu.get_device(None)
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.classes = list(range(self.num_classes))
        self.delta = None
        self.minimal = minimal # early stop + no eps
        self.project = not self.minimal
        self.loss = -np.inf

    def evaluate_attack(self, dl, model):
        model = model.to(self.device)
        Xa, Ya, Yh, P = [], [], [], []

        for xb, yb in dl:
            xb, yb = xb.to(self.device), yb.to(self.device)
            delta = self.perturb(xb, yb, model)
            xba = xb+delta
            
            with torch.no_grad():
                out = model(xba).detach()
            yh = torch.argmax(out, dim=1)
            xb, yb, yh, xba, delta = xb.cpu(), yb.cpu(), yh.cpu(), xba.cpu(), delta.cpu()
            
            Ya.append(yb)
            Yh.append(yh)
            Xa.append(xba)
            P.append(delta)

        Xa, Ya, Yh, P = map(torch.cat, [Xa, Ya, Yh, P])
        ta_dl = utils._to_dl(Xa, Ya, dl.batch_size)
        acc, loss = utils.compute_loss_and_accuracy_from_dl(ta_dl, model,
                                                            F.cross_entropy,
                                                            device=self.device)
        return {
            'acc': acc.item(),
            'loss': loss.item(),
            'ta_dl': ta_dl,
            'Xa': Xa.numpy(),
            'Ya': Ya.numpy(),
            'Yh': Yh.numpy(),
            'P': P.numpy()
        }

    def perturb(self, xb, yb, model, cpu=False):
        model, xb, yb = model.to(self.device), xb.to(self.device), yb.to(self.device)
        if self.eps == 0: return torch.zeros_like(xb)

        # compute perturbations and track best perturbations
        self.loss = -np.inf
        max_delta = self._perturb_once(xb, yb, model)
        
        with torch.no_grad(): 
            out = model(xb+max_delta)
            max_loss = nn.CrossEntropyLoss(reduction='none')(out, yb)

        for _ in range(self.restarts-1):
            delta = self._perturb_once(xb, yb, model)

            with torch.no_grad():
                out = model(xb+delta)
                all_loss = nn.CrossEntropyLoss(reduction='none')(out, yb)

            loss_flag = all_loss >= max_loss
            max_delta[loss_flag] = delta[loss_flag]
            max_loss = torch.max(max_loss, all_loss)

        if cpu: max_delta = max_delta.cpu()
        return max_delta

    def _perturb_once(self, xb, yb, model, track_scores=False, stop_const=1e-5):
        self.delta = self._init_delta(xb, yb)
        scores = []

        # (minimal) mask perturbations if model already misclassifies
        for t in range(self.num_iter):
            loss, out = self._get_loss(xb, yb, model, get_scores=True)
    
            if self.minimal:
                yh = torch.argmax(out, dim=1).detach()
                not_flipped = yh == yb            
                not_flipped_ratio = not_flipped.sum().item()/float(len(yb))
            else:
                not_flipped = None
                not_flipped_ratio = 1.0

            # stop if almost all examples in the batch misclassified    
            if not_flipped_ratio < stop_const: 
                break
            
            if track_scores: 
                scores.append(out.detach().cpu().numpy())
                
            # compute loss, update + clamp delta
            loss.backward()
            self.loss = max(self.loss, loss.item())

            self.delta = self._update_delta(xb, yb, update_mask=not_flipped)
            self.delta = self._clamp_input(xb, yb)
            
        d = self.delta.detach()

        if track_scores: 
            scores = np.stack(scores).swapaxes(0, 1)
            return d, scores
        
        return d

    def _init_delta(self, xb, yb):
        delta = torch.empty_like(xb)
        delta = delta.uniform_(-self.rand_eps, self.rand_eps)
        delta = delta.to(self.device)
        delta.requires_grad = True
        return delta

    def _clamp_input(self, xb, yb):
        # clamp delta s.t. X+delta in valid input range
        self.delta.data = torch.max(self.B[0]-xb,
                                    torch.min(self.B[1]-xb,
                                              self.delta.data))
        return self.delta

    def _get_loss(self, xb, yb, model, get_scores=False):
        out = model(xb+self.delta)

        if self.loss_type == 'untargeted':
            L = -1*F.cross_entropy(out, yb)

        elif self.loss_type == 'targeted':
            L = nn.CrossEntropyLoss()(out, yb)

        elif self.loss_type == 'random_targeted':
            rand_yb = torch.randint(low=0, high=self.num_classes, size=(len(yb),), device=self.device)
            #rand_yb[rand_yb==yb] = (yb[rand_yb==yb]+1) % self.num_classes
            L = nn.CrossEntropyLoss()(out, rand_yb)

        elif self.loss_type == 'plusone_targeted':
            next_yb = (yb+1) % self.num_classes
            L = nn.CrossEntropyLoss()(out, next_yb)

        elif self.loss_type == 'binary_targeted':
            yb_opp = 1-yb
            L = nn.CrossEntropyLoss()(out, yb_opp)

        elif self.loss_type ==  'binary_hybrid':
            yb_opp = 1-yb
            L = nn.CrossEntropyLoss()(out, yb_opp) - nn.CrossEntropyLoss()(out, yb)

        else:
            assert False, "unknown loss type"

        if get_scores: return L, out
        return L 

class L2_PGD_Attack(PGD_Attack):

    OVERFLOW_CONST = 1e-10

    def get_norms(self, X):
        nch = len(X.shape)
        return X.view(X.shape[0], -1).norm(dim=1)[(...,) + (None,)*(nch-1)]

    def _update_delta(self, xb, yb, update_mask=None):
        # normalize gradients
        grad = self.delta.grad.detach()
        norms = self.get_norms(grad)
        grad = grad/(norms+self.OVERFLOW_CONST) # add const to avoid overflow

        # steepest descent
        if self.minimal and update_mask is not None:
            um = update_mask
            self.delta.data[um] = self.delta.data[um] - self.lr*grad[um]
        else:
            self.delta.data = self.delta.data - self.lr*grad

        # l2 ball projection
        if self.project:
            delta_norms = self.get_norms(self.delta.data)
            self.delta.data = self.eps*self.delta.data / (delta_norms.clamp(min=self.eps))

        self.delta.grad.zero_()
        return self.delta

    def _init_delta(self, xb, yb):
        # random vector with L2 norm rand_eps
        delta = torch.zeros_like(xb)
        delta = delta.uniform_(-self.rand_eps, self.rand_eps)
        delta_norm = self.get_norms(delta)
        delta = self.rand_eps*delta/(delta_norm+self.OVERFLOW_CONST)
        delta = delta.to(self.device)
        delta.requires_grad = True
        return delta

class Linf_PGD_Attack(PGD_Attack):

    def _update_delta(self, xb, yb, **kw):
        # steepest descent + linf projection (GD)
        self.delta.data = self.delta.data - self.lr*(self.delta.grad.detach().sign())
        self.delta.data = self.delta.data.clamp(-self.eps, self.eps)
        self.delta.grad.zero_()
        return self.delta

    
# UAP methods

class AS_UAP(object):
    """
    UAP method (Algorithm 2) in Universal Adversarial Training paper (https://arxiv.org/abs/1811.11304)
    not using clipped version to avoid hyper-parameter tuning
    (even with tuning, improvement is marginal)
    """

    def __init__(self, eps, lr, num_iter, shape, num_classes=2, bounds=(0.,1.),
                 loss_type='untargeted', rand_eps=0., device=None):
        self.device = device if device else gu.get_device(None)
        self.loss_type = loss_type
        self.B = bounds
        self.rand_eps = rand_eps
        self.eps = eps
        self.lr = lr
        self.num_iter = num_iter
        self.num_classes = num_classes
        self.classes = list(range(self.num_classes))
        self.shape = shape
        self._init_delta()
        
    @property
    def uap(self):
        return copy.deepcopy(self.delta.detach().cpu()).numpy()

    def fit(self, dl, model, num_epochs):
        # compute uap
        model = model.to(self.device)
        for t in range(num_epochs):
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)

                # update + project + clamp delta t times
                for t in range(self.num_iter):
                    loss = self._get_loss(xb, yb, model, True)
                    loss.backward()
                    self.delta = self._update_delta(xb, yb)

    def evaluate_attack(self, dl, model, **kw):
        model = model.to(self.device)
        X, Xa, Ya, P = [], [], [], []

        for xb, yb in dl:
            xb, yb = xb.to(self.device), yb.to(self.device)
            xba, ptb = self._perturb(xb, yb, False)
            X.append(xb.cpu())
            Xa.append(xba.cpu())
            Ya.append(yb.cpu())
            P.append(ptb.cpu())

        X, Xa, Ya, P = map(torch.cat, [X, Xa, Ya, P])
        ta_dl = utils._to_dl(Xa, Ya, dl.batch_size)
        acc_func = utils.compute_loss_and_accuracy_from_dl
        acc, loss = acc_func(ta_dl, model, F.cross_entropy, device=self.device)

        return {
            'P': P,
            'X': X,
            'Xa': Xa,
            'Ya': Ya,
            'acc': acc.item(),
            'loss': loss.item(),
            'dl': ta_dl
        }

    def _get_loss(self, xb, yb, model, train_mode):
        xba, delta = self._perturb(xb, yb, train_mode=train_mode)
        out = model(xba)

        if self.loss_type == 'untargeted':
            return -1*nn.CrossEntropyLoss()(out, yb)

        elif self.loss_type == 'targeted':
            return nn.CrossEntropyLoss()(out, yb)

        elif self.loss_type == 'binary_targeted':
            yb_opp = 1-yb
            return nn.CrossEntropyLoss()(out, yb_opp)

        elif self.loss_type ==  'binary_hybrid':
            yb_opp = 1-yb
            return nn.CrossEntropyLoss()(out, yb_opp) - nn.CrossEntropyLoss()(out, yb)

        else:
            assert False, "unknown loss type"

    def _perturb(self, xb, yb, train_mode):
        # broadcast clamped + scaled + signed-if-binary UAPs
        d = self.delta if train_mode else self.delta.data
        sign = ((2*yb-1)*1.0) if self.num_classes == 2 else torch.ones(len(yb)).to(self.device)
        sign = sign[(...,)+(None,)*(len(xb.shape)-1)].float()
        delta = torch.zeros_like(xb, device=self.device)
        delta = sign*(delta+d)

        # perturb and re-clamp data
        xba = (xb + delta).clamp(self.B[0], self.B[1])
        delta = xba-xb
        return xba, delta

    def _init_delta(self):
        delta = torch.zeros(*self.shape)
        delta = delta.uniform_(-self.rand_eps, self.rand_eps).to(self.device)
        delta.requires_grad = True
        self.delta = delta

class Linf_AS_UAP(AS_UAP):

    def _update_delta(self, xb, yb):
        # steepest descent + linf projection (GD)
        self.delta.data = self.delta.data - self.lr*(self.delta.grad.detach().sign())
        self.delta.data = self.delta.data.clamp(-self.eps, self.eps)
        self.delta.grad.zero_()
        return self.delta

class L2_AS_UAP(AS_UAP):

    OVERFLOW_CONST = 1e-10

    def _update_delta(self, xb, yb):
        # normalize gradients
        grad = self.delta.grad.detach()
        norms = grad.norm()
        grad = grad/(norms+self.OVERFLOW_CONST) # add const to avoid overflow

        # steepest descent
        self.delta.data = self.delta.data - self.lr*grad

        # l2 ball projection
        delta_norms = self.delta.data.norm()
        self.delta.data = self.eps*self.delta.data / (delta_norms.clamp(min=self.eps))

        self.delta.grad.zero_()
        return self.delta


class SVD_UAP(object):
    # based on https://arxiv.org/abs/2005.08632

    def __init__(self, attack, bounds=(0.,1.), device=None, num_classes=2):
        self.device = device if device else gu.get_device(None)
        self.attack = attack
        self.attack.device = self.device
        self.B = bounds
        self.num_classes = num_classes

    def fit(self, dl, model):
        # get perturbations
        model = model.to(self.device)
        pdata = self.attack.evaluate_attack(dl, model)

        self.p_acc = pdata['acc']
        shape = list(pdata['Xa'].shape)
        self.num_imgs, self.img_shape = shape[0], shape[1:]

        # run SVD
        P = pdata['P'].reshape(pdata['P'].shape[0], -1)
        self.P = P / np.linalg.norm(P, axis=1)[:, None]
        U, self.S, self.VH = np.linalg.svd(P)
        del U

        # setup UAP
        self.uaps = self.VH.reshape(self.VH.shape[0], *self.img_shape)
        self.uap = self.uaps[0]

    def evaluate_attack(self, dl, model, eps, kth=0, **kw):
        self.delta = torch.FloatTensor(self.uaps[kth]).to(self.device)
        eval1 = self._eval(dl, model, eps, 1.0)
        eval2 = self._eval(dl, model, eps, -1.0)
        if eval1['acc'] < eval2['acc']: return eval1
        return eval2

    def _eval(self, dl, model, eps, pos_dir):
        model = model.to(self.device)
        X, Xa, Ya, P = [], [], [], []

        for xb, yb in dl:
            xb, yb = xb.to(self.device), yb.to(self.device)
            xba, ptb = self._perturb(xb, yb, eps, pos_dir)
            X.append(xb.cpu())
            Xa.append(xba.cpu())
            Ya.append(yb.cpu())
            P.append(ptb.cpu())

        X, Xa, Ya, P = map(torch.cat, [X, Xa, Ya, P])
        ta_dl = utils._to_dl(Xa, Ya, dl.batch_size)
        acc_func = utils.compute_loss_and_accuracy_from_dl
        acc, loss = acc_func(ta_dl, model, F.cross_entropy, device=self.device)

        return {
            'P': P,
            'X': X,
            'Xa': Xa,
            'Ya': Ya,
            'acc': acc.item(),
            'loss': loss.item(),
            'dl': ta_dl,
            'pos_dir': pos_dir
        }

    def _perturb(self, xb, yb, eps, pos_dir):
        nch = len(xb.shape)

        # broadcast clamped + scaled + signed UAPs
        sign = ((2*yb-1)*(1.0)) if self.num_classes == 2 else torch.ones(len(yb)).to(self.device)
        sign = sign[(...,)+(None,)*(len(xb.shape)-1)].float()
        
        delta = torch.zeros_like(xb, device=self.device)
        delta = eps*pos_dir*sign*(delta + self.delta)

        # perturb and re-clamp data
        xba = xb + delta
        xba = xba.clamp(self.B[0], self.B[1])
        delta = xba-xb
        return xba, delta

