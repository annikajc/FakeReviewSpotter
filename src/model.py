import torch
import pytorch_lightning as pl
import sys
import torchmetrics
sys.path.append("./fairseq")
from fairseq.data.data_utils import collate_tokens


class FakeReviewsRoberta(torch.nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        
        # load pre-trained Roberta model and set to train mode
        self.roberta = torch.hub.load('facebookresearch/fairseq', 'roberta.base')

        # set to training mode
        self.roberta.train()
        self.roberta.cuda()
        self.roberta.register_classification_head('fake_reviews', pooler_activation_fn=torch.nn.ReLU(), pooler_dropout=torch.nn.Dropout(p=0.5), num_classes=num_classes)
    
    def forward(self, batch):
        tokens = collate_tokens([self.roberta.encode(review) for review in batch], pad_idx=1)
        x = self.roberta.predict('fake_reviews', tokens)
        return x
    

class FakeReviewsLightning(pl.LightningModule):
    def __init__(self, num_classes=1, clearml_logger=None):
        super().__init__()

        # initialize logger
        self.clearml_logger = clearml_logger

        # initialize model
        self.model = FakeReviewsRoberta(num_classes=num_classes)

        # initialize loss function
        self.loss_fn = torch.nn.BCELoss()

        # initialize model metrics
        # accuracy
        self.accuracy = torchmetrics.Accuracy(task="binary")

        # precision
        self.precision = torchmetrics.Precision(task="binary")

        # recall
        self.recall = torchmetrics.Recall(task="binary")

        # F1-score
        self.f1 = torchmetrics.F1Score(task="binary")

        # Confusion Matrix
        self.cm = torchmetrics.ConfusionMatrix(task="binary")

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        # initialize Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-02, total_steps=int((40000*0.9*0.8/8)+1)*300)
        
        return [optimizer], [lr_scheduler]

    def on_train_start(self):
        # self.model.roberta.train()
        self.model.roberta.eval()

    def on_validation_start(self):
        self.model.roberta.eval()

    def on_test_start(self):
        self.model.roberta.eval()


    def training_step(self, batch, batch_idx):
        # get reviews and labels from batch
        reviews, labels = batch

        # pass reviews through model
        outputs = self.forward(reviews)
        labels.resize_(outputs.size(dim=0), 1)
        loss = self.loss_fn(outputs, labels.to(torch.float32))

        # metrics
        accuracy = self.accuracy(outputs, labels)
        precision = self.precision(outputs, labels)
        recall = self.recall(outputs, labels)
        f1 = self.f1(outputs, labels)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(
            {'train_accuracy': accuracy, 'train_precision': precision, 'train_recall': recall, 'train_f1': f1},
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        # get reviews and labels from batch
        reviews, labels = batch
        
        # pass reviews through model
        outputs = self.forward(reviews)
        labels.resize_(outputs.size(dim=0), 1)
        loss = self.loss_fn(outputs, labels.to(torch.float32))

        # metrics
        accuracy = self.accuracy(outputs, labels)
        precision = self.precision(outputs, labels)
        recall = self.recall(outputs, labels)
        f1 = self.f1(outputs, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(
            {'val_accuracy': accuracy, 'val_precision': precision, 'val_recall': recall, 'val_f1': f1},
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        # get reviews and labels from batch
        reviews, labels = batch
        labels.resize_(reviews.size(dim=0), 1)

        # pass reviews through model
        outputs = self.forward(reviews)
        labels.resize_(outputs.size(dim=0), 1)
        loss = self.loss_fn(outputs, labels.to(torch.float32))

        # metrics
        accuracy = self.accuracy(outputs, labels)
        precision = self.precision(outputs, labels)
        recall = self.recall(outputs, labels)
        f1 = self.f1(outputs, labels)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(
            {'test_accuracy': accuracy, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1},
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        return {'loss': loss}


