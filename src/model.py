import torch
import pytorch_lightning as pl
import sys
from transformers import AutoTokenizer, RobertaForSequenceClassification
from torchmetrics.classification import BinaryConfusionMatrix

class FakeReviewsRoberta(torch.nn.Module):
    def __init__(self, num_classes = 1, stage = "train", device=None, size=None):
        super().__init__()
        if size == None:
            size = 12
        # load pre-trained Roberta model and set to train mode
        self.roberta = RobertaForSequenceClassification.from_pretrained("roberta-base",
                                                                        num_labels=num_classes,
                                                                        num_hidden_layers=size,
                                                                        num_attention_heads=size,
                                                                        ignore_mismatched_sizes=True).to(device)
        self.stage = stage
        self.device = device

    def forward(self, ids, attention_mask, targets):
        if self.stage == "train":
            outputs = self.roberta(ids, attention_mask=attention_mask, labels=targets)
        else:
            with torch.no_grad():
                outputs = self.roberta(ids, attention_mask=attention_mask, labels=targets)
        return outputs.logits
    

class FakeReviewsLightning(pl.LightningModule):
    def __init__(self, num_classes=1, clearml_logger=None, size=12, device = None):
        super().__init__()

        # initialize logger
        self.clearml_logger = clearml_logger

        # initialize model
        self.model = FakeReviewsRoberta(num_classes=num_classes, size=size, device=device)

        # initialize loss function
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        # initialize model metrics for training and validation
        self.train_cm = BinaryConfusionMatrix()
        self.val_cm = BinaryConfusionMatrix()
        self.test_cm = BinaryConfusionMatrix()

    def forward(self, ids, mask, targets):
        return self.model(ids, attention_mask=mask, targets=targets)
    
    def configure_optimizers(self):
        # initialize Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}

    def on_train_start(self):
        # self.model.roberta.train()
        self.model.roberta.stage = "train"
 
    def on_validation_start(self):
        self.model.roberta.stage = "validation"

    def on_test_start(self):
        self.model.roberta.stage = "test"

    def on_predict_start(self):
        self.model.roberta.stage = "predict"

    def training_step(self, batch, batch_idx):
        # get reviews and labels from batch
        ids = batch['ids']
        mask = batch['mask']
        targets = batch['targets']

        # pass reviews through model
        outputs = self.forward(ids, mask, targets)
        targets.resize_(outputs.size(dim=0), 1)
        loss = self.loss_fn(outputs, targets.to(torch.float))

        # metrics
        self.train_cm.update(outputs, targets)

        self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}

    def on_training_epoch_end(self):
        # compute training metrics and log them
        epoch_train_cm = self.train_cm.compute()
        accuracy = (epoch_train_cm[0][0] + epoch_train_cm[1][1]) / (epoch_train_cm[0][0] + epoch_train_cm[0][1] + epoch_train_cm[1][0] + epoch_train_cm[1][1])
        precision = (epoch_train_cm[1][1]) / (epoch_train_cm[1][1] + epoch_train_cm[0][1]) if (epoch_train_cm[1][1] + epoch_train_cm[0][1]) > 0 else 0
        recall = (epoch_train_cm[1][1]) / (epoch_train_cm[1][1] + epoch_train_cm[1][0]) if (epoch_train_cm[1][1] + epoch_train_cm[1][0]) > 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

        self.log_dict({
                'train_accuracy': accuracy,
                'train_precision': precision,
                'train_recall': recall,
                'train_f1': f1
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.train_cm.reset()

    def validation_step(self, batch, batch_idx):
        # get reviews and labels from batch
        ids = batch['ids']
        mask = batch['mask']
        targets = batch['targets']
        
        # pass reviews through model
        with torch.no_grad():
            outputs = self.forward(ids, mask, targets)
        targets.resize_(outputs.size(dim=0), 1)
        loss = self.loss_fn(outputs, targets.to(torch.float))

        # metrics
        self.val_cm.update(outputs, targets)

        self.log('val_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}
    
    def on_validation_epoch_end(self):
        epoch_val_cm = self.val_cm.compute()
      
        accuracy = (epoch_val_cm[0][0] + epoch_val_cm[1][1]) / (epoch_val_cm[0][0] + epoch_val_cm[0][1] + epoch_val_cm[1][0] + epoch_val_cm[1][1])
        precision = (epoch_val_cm[1][1]) / (epoch_val_cm[1][1] + epoch_val_cm[0][1]) if (epoch_val_cm[1][1] + epoch_val_cm[0][1]) > 0 else 0
        recall = (epoch_val_cm[1][1]) / (epoch_val_cm[1][1] + epoch_val_cm[1][0]) if (epoch_val_cm[1][1] + epoch_val_cm[1][0]) > 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

        if self.global_step > 0:
            self.log_dict({
                'val_accuracy': accuracy,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.val_cm.reset()
    
    def test_step(self, batch, batch_idx):
        # get reviews and labels from batch
        ids = batch['ids']
        mask = batch['mask']
        targets = batch['targets']

        # pass reviews through model
        with torch.no_grad():
            outputs = self.forward(ids, mask, targets)
        targets.resize_(outputs.size(dim=0), 1)
        loss = self.loss_fn(outputs, targets.to(torch.float))

        # metrics
        self.test_cm.update(outputs, targets)

        self.log('test_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}

    def on_test_epoch_end(self):
        epoch_test_cm = self.test_cm.compute()
      
        accuracy = (epoch_test_cm[0][0] + epoch_test_cm[1][1]) / (epoch_test_cm[0][0] + epoch_test_cm[0][1] + epoch_test_cm[1][0] + epoch_test_cm[1][1])
        precision = (epoch_test_cm[1][1]) / (epoch_test_cm[1][1] + epoch_test_cm[0][1]) if (epoch_test_cm[1][1] + epoch_test_cm[0][1]) > 0 else 0
        recall = (epoch_test_cm[1][1]) / (epoch_test_cm[1][1] + epoch_test_cm[1][0]) if (epoch_test_cm[1][1] + epoch_test_cm[1][0]) > 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

        self.log_dict({
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.test_cm.reset()
    