from src.dataset import DataModuleFakeReviews
from src.model import FakeReviewsLightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning import Trainer
from clearml import Task
from transformers import AutoTokenizer

def main():
    # create ClearML task
    task = Task.init(project_name="Fake Review Detection",
                     task_name="transfer RoBERTa - reduced size")

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

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    datamodule = DataModuleFakeReviews(batch_size=2, tokenizer=tokenizer, max_length=256, num_workers=2)

    model = FakeReviewsLightning(clearml_logger=task.get_logger(), device=device).to(device)

    # initialize checkpoint callback depending on parse arguments
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints_reduced/", monitor="val_loss", mode="min")

    trainer = Trainer(accelerator="gpu", max_epochs=300, profiler="simple",
                          callbacks=[checkpoint_callback, ModelSummary(4),
                                     EarlyStopping(monitor="val_loss",
                                                   mode="min",
                                                   patience=10)],
                          strategy="auto", enable_checkpointing=True)
    # else:
    #     print("No GPU")

    # fit model
    trainer.fit(model=model, datamodule=datamodule)
    

if __name__ == '__main__':
    main()
