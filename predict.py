import torch
from src.model import FakeReviewsLightning
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

def is_fake(reviews):
    """
    Helper function for reading in the reviews passed in as input
    and outputting the prediction.
    """
    # get device
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    # load model weights and pretrained tokenizer
    model = FakeReviewsLightning.load_from_checkpoint('./checkpoints_reduced/reduced.ckpt', size=4, device=device).to(device)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    for review in reviews:
        title = str(review)
        title = " ".join(title.split())
        inputs = tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            padding='max_length',
            max_length=256,
            return_token_type_ids=True,
            truncation=True,
            return_tensors='pt'
        )
        ids = inputs['input_ids'].to(dtype=torch.long).flatten().to(device)
        mask = inputs['attention_mask'].to(dtype=torch.long).flatten().to(device)
        prediction = model(ids.unsqueeze(dim=0), mask.unsqueeze(dim=0), None)
        fake = 0 if torch.nn.functional.sigmoid(prediction).item() < 0.5 else 1
        # Review is real
        if fake == 0:
            return False
        # Review is fake
        else:
            return True