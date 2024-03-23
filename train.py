from src.dataset import DataModuleFakeReviews
from src.model import FakeReviewsLightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, EarlyStopping
from pytorch_lightning import Trainer

def main():
    # initialize data module and model
    print("mreeow")
    print("   \   ")
    print("   ^ ^ ")
    print("  {'.'}")
    print("? /  | ")
    print("\|  || ")

    # check if cuda is available
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(device)

    # enable cudnn benchmarking if available
    torch.backends.cudnn.benchmark = True if torch.backends.cudnn.is_available() and cuda_available else False
    precision = '16-mixed'

    torch.set_float32_matmul_precision('high')

    datamodule = DataModuleFakeReviews(batch_size=8)
    model = FakeReviewsLightning().to(device)

    # initialize checkpoint callback depending on parse arguments
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/", monitor="loss", mode="min")

    trainer = Trainer(accelerator="gpu", max_epochs=300, profiler="simple",
                          callbacks=[checkpoint_callback, ModelSummary(4),
                                     EarlyStopping(monitor="loss",
                                                   mode="min",
                                                   patience=10)],
                          strategy="auto", enable_checkpointing=True, precision=precision)
    
    # else:
    #     print("No GPU")

    # fit model
    trainer.fit(model=model, datamodule=datamodule)
    

if __name__ == '__main__':
    main()