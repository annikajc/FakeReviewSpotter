from src.dataset import DataModuleFakeReviews
from src.model import FakeReviewsLightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning import Trainer
from lightning.pytorch.tuner import Tuner
from clearml import Task

def main():
    # create ClearML task
    task = Task.init(project_name="Fake Review Detection",
                     task_name="finetune RoBERTa - v1.0")

    # initialize data module and model
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(device)
    print("   \   ")
    print("   ^ ^ ")
    print("  {'.'}")
    print("? /  | ")
    print("\|  || ")

    # enable cudnn benchmarking if available
    torch.backends.cudnn.benchmark = True if torch.backends.cudnn.is_available() and cuda_available else False

    torch.set_float32_matmul_precision('high')

    datamodule = DataModuleFakeReviews(batch_size=1, num_workers=4)

    model = FakeReviewsLightning(clearml_logger=task.get_logger(), device=device).to(device)

    # initialize checkpoint callback depending on parse arguments
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/", monitor="val_loss", mode="min")

    trainer = Trainer(accelerator="gpu", max_epochs=1, profiler="simple",
                          callbacks=[checkpoint_callback, ModelSummary(4),
                                     EarlyStopping(monitor="val_loss",
                                                   mode="min",
                                                   patience=10)],
                          strategy="auto", enable_checkpointing=True, limit_train_batches=50, limit_val_batches=50)
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, datamodule=datamodule, mode="power")
    
    # else:
    #     print("No GPU")

    # fit model
    trainer.fit(model=model, datamodule=datamodule)
    

if __name__ == '__main__':
    main()
