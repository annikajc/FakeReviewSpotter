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
                     task_name="Test RoBERTa on Yelp Dataset")

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

    datamodule = DataModuleFakeReviews(batch_size=2, tokenizer=tokenizer, max_length=256, dataset="yelp", num_workers=0)

    model = FakeReviewsLightning.load_from_checkpoint("./checkpoints_reduced/reduced.ckpt", size=4, clearml_logger=task.get_logger(), device=device).to(device)

    trainer = Trainer(accelerator="gpu", max_epochs=300, profiler="simple",
                          callbacks=[ModelSummary(4)],
                          strategy="auto", enable_checkpointing=False)
    # else:
    #     print("No GPU")

    # fit model
    trainer.test(model=model, datamodule=datamodule)
    

if __name__ == '__main__':
    main()
