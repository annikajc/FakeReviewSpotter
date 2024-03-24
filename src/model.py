import torch
import pytorch_lightning as pl
import sys
import torchmetrics
sys.path.append("./fairseq")
from fairseq.data.data_utils import collate_tokens
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

class FakeReviewsRoberta(torch.nn.Module):
    def __init__(self, num_classes=1, device="cuda"):
        super().__init__()
        
        # load pre-trained Roberta model and set to train mode
        self.roberta = torch.hub.load('facebookresearch/fairseq', 'roberta.base')

        # set to training mode
        self.roberta.train()
        self.roberta.cuda()
        self.roberta.register_classification_head('fake_reviews', pooler_activation_fn=torch.nn.ReLU(), pooler_dropout=torch.nn.Dropout(p=0.5), num_classes=num_classes)

        # decoder layers
        # self.dropout = torch.nn.Dropout(p=0.5).to(device)
        # self.dense = torch.nn.Linear(self.roberta.model.args.encoder_embed_dim, self.roberta.model.args.encoder_embed_dim).to(device)
        # self.activation_fn = torch.nn.ReLU().to(device)
        # self.out = torch.nn.Linear(self.roberta.model.args.encoder_embed_dim, num_classes).to(device)

    def forward(self, batch):
        x = collate_tokens([self.roberta.encode(review) for review in batch], pad_idx=1)
        x = self.roberta.predict('fake_reviews', x)
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

        # initialize model metrics for training and validation
        self.train_metrics = MetricCollection({
            'train_accuracy': BinaryAccuracy()
        })
#, BinaryPrecision(),
#            BinaryRecall(), BinaryF1Score()
      #  ])
        self.val_metrics = MetricCollection({
            'val_accuracy': BinaryAccuracy()
        })
#, BinaryPrecision(),
#            BinaryRecall(), BinaryF1Score()
#        ])

        self.validation_epoch_loss = []

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        # initialize Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-02, total_steps=int((40000*0.9*0.8)+1)*300)
        
        return [optimizer], [lr_scheduler]

    def on_train_start(self):
        # self.model.roberta.train()
        self.model.roberta.train()
 
    def on_validation_start(self):
        self.model.roberta.eval()

    def on_test_start(self):
        self.model.roberta.eval()


    def training_step(self, batch, batch_idx):
        # get reviews and labels from batch
        reviews, labels = batch

        # pass reviews through model
        outputs = self.forward(reviews)
        if batch_idx < 10:
            print(outputs, labels)
        labels.resize_(outputs.size(dim=0), 1)
        loss = self.loss_fn(outputs, labels.to(torch.float))

        # metrics
       	self.train_metrics.update(outputs, labels)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}

    def on_training_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        # get reviews and labels from batch
        reviews, labels = batch
        
        # pass reviews through model
        outputs = self.forward(reviews)
        labels.resize_(outputs.size(dim=0), 1)
        loss = self.loss_fn(outputs, labels.to(torch.float))

        # metrics
        self.val_metrics.update(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.validation_epoch_loss.append(loss)
        return {'loss': loss}
    
    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()

        avg_loss = sum(self.validation_epoch_loss) / len(self.validation_epoch_loss)
        self.clearml_logger.report_scalar(
                title='val_loss', series='val_loss', value=avg_loss, iteration=self.global_step
        )
        self.validation_epoch_loss = []
    
    def test_step(self, batch, batch_idx):
        # get reviews and labels from batch
        reviews, labels = batch
        labels.resize_(reviews.size(dim=0), 1)

        # pass reviews through model
        outputs = self.forward(reviews)
        labels.resize_(outputs.size(dim=0), 1)
        loss = self.loss_fn(outputs, labels.to(torch.float))

        # metrics
        self.accuracy(outputs, labels)
        self.precision(outputs, labels)
        self.recall(outputs, labels)
        self.f1(outputs, labels)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(
            {'test_accuracy': self.accuracy, 'test_precision': self.precision, 'test_recall': self.recall, 'test_f1': self.f1},
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        return {'loss': loss}


