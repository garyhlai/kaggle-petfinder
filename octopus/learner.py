from typing import Set, List, Dict
import torch
from torch.cuda import amp
from fastai.basics import store_attr, noop
import dill

class Learner:
    def __init__(self, model, dls, loss_func, optimizer, lr, valid_interval=2, cbs=[], get_pred=None):
        store_attr('model, dls, loss_func, optimizer, lr, valid_interval, cbs, get_pred', self)
        if get_pred == None: self.get_pred = self.model
        for cb in cbs: cb.learner = self
        self.cb_dict = { type(cb).__name__: cb for cb in self.cbs}

    def fit(self, epochs):
        self.epochs = epochs
        self('before_fit')      
        for epoch in range(1, epochs+1):
            self.epoch = epoch
            self('before_epoch')
            self.model.train()
            for i, batch in enumerate(self.dls.train_dl):
              self.batch_x, self.batch_y = batch
              self.train_batch()
              if i % self.valid_interval == 0:
                  self.validate_interval()
            self('after_epoch')
        self('after_fit')
    
    def validate_interval(self):
        self.validate_model() 
        self.model.train()
    
    def train_batch(self):
        self('before_train_batch')
        # forward
        self.pred = self.get_pred(self.batch_x, self.model)
        self.loss = self.loss_func(self.pred, self.batch_y)
        self('after_train_loss')
        # backward
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self('after_train_batch')
    
    def validate_model(self):
        self('before_validate')
        self.model.eval()
        for batch_x, batch_y in self.dls.valid_dl:
            self.batch_x, self.batch_y = batch_x, batch_y
            self.validate_batch()

        self('after_validate')

    def validate_batch(self):
        self('before_valid_batch')
        # forward
        with torch.no_grad(): 
            self.pred = self.get_pred(self.batch_x, self.model)
            self.loss = self.loss_func(self.pred, self.batch_y)
        self('after_valid_loss')

    def __call__(self,name):
        for cb in self.cbs: getattr(cb,name,noop)()

    def export_attr_from_cb(self, cb, attrs: Set[str]):
      return {key: cb.__dict__[key] for key in attrs}

    def export(self, export_attrs: Dict[str, Set[str]]):
      export = {}
      for cb_name, attrs in export_attrs.items():
        cb = self.cb_dict[cb_name]
        export[cb_name] = self.export_attr_from_cb(cb, attrs)
      return export 

    def save(self, export_attrs, path):
      export = self.export(export_attrs)
      with open(f"{path}/mylearner.pkl", 'wb') as learner_file:
        dill.dump(export, learner_file)
        print("save successful!")

class MixedPrecisionLearner(Learner):
    def fit(self, epochs):
        self.epochs = epochs
        self.scaler = amp.GradScaler()

        self('before_fit')      
        for epoch in range(1, epochs+1):
            self.epoch = epoch
            self('before_epoch')
            self.model.train()
            for i, batch in enumerate(self.dls.train_dl):
              self.batch_x, self.batch_y = batch
              self.train_batch()
              if i % self.valid_interval == 0:
                  self.validate_interval()
            self('after_epoch')
        self('after_fit')

    def train_batch(self):
        self('before_train_batch')
        # forward
        with amp.autocast():
            self.pred = self.get_pred(self.batch_x, self.model)
            self.loss = self.loss_func(self.pred, self.batch_y)
        self('after_train_loss')
        # backward
        self.scaler.scale(self.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self('after_train_batch')

    def validate_batch(self):
        self('before_valid_batch')
        # forward
        with torch.no_grad(): 
          with amp.autocast():
            self.pred = self.get_pred(self.batch_x, self.model)
            self.loss = self.loss_func(self.pred, self.batch_y)
        self('after_valid_loss')