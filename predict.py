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
    input = ["I recently finished reading a novel from my favorite author, and I must say, it was absolutely captivating. The characters were so well-developed, and the plot twists kept me on the edge of my seat until the very end. I highly recommend it to anyone looking for a thrilling read!",
             "I purchased a pair of shoes online, and when they arrived, I was pleasantly surprised by their quality. They fit perfectly and are incredibly comfortable to wear all day. Plus, the color goes well with almost any outfit. I couldn't be happier with my purchase!"]
    predict(input)