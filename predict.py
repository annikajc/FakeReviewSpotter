import torch
from src.model import FakeReviewsLightning
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

def predict(reviews):
    """
    Helper function for reading in the reviews passed in as input
    and outputting the prediction.
    """
    # get device
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    # load model weights and pretrained tokenizer
    model = FakeReviewsLightning.load_from_checkpoint('./checkpoints/amazon_weights.ckpt', device=device).to(device)
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
        print(prediction)
        fake = 0 if torch.nn.functional.sigmoid(prediction).item() < 0.5 else 1
        if fake == 0:
            print('Review is real.')
        else:
            print('Review is fake.')

     

if __name__ == '__main__':
    input = ["I switched over from Nike. Best decision ever. These shoes last longer, provide a more solid foundation, and are more comfortable than any other brand I’ve tried.",
             "These shoes seem like they could be good court shoes. Unfortunately I need size 14.5 in this shoe and that size is not offered. Had to buy size 15 forcing me to tie them tight enough to hurt the top of my feet because if they are looser my feet slide around and I have already lost 2 toe nails. They may work well for people that are able to get a good fit. They seem well made."]
    predict(input)