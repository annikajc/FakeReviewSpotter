import gradio as gr
from predict import is_fake
import style

def predict_review(review):
    """
    Print output for review analysis
    """
    if is_fake(review):
        return "<h2 class='result-fake'>Review is fake.</h2>"
    else:
        return "<h2 class='result-real'>Review is real.</h2>"


def predict_link(link):
    """
    Output for Amazon link analysis
    """
    # TODO: CHANGE DUMMY VALUES TO VARIABLES + ADD FUNCTIONALITY
    result = "<h2>Fake Reviews:</h2>"
    result += style.percent_color(30)
    
    result += "<h2>Adjusted Rating:</h2>"
    result += style.rating_color(2.0)

    return result


with gr.Blocks(css=style.css) as demo:
    gr.Markdown("# Analyze fake reviews here!")
    gr.Markdown("Determine if a review is fake, or assess the fake review rate of any Amazon product.")

    with gr.Tab("Analyze Review"):
        review = gr.Textbox(label="Insert review")
        review_btn = gr.Button("Analyze")
        review_result = gr.HTML(label="Result")
        review_btn.click(fn=predict_review, inputs=review, outputs=review_result, api_name="predict_review")

    with gr.Tab("Analyze Link"):
        link = gr.Textbox(label="Insert Amazon link")
        link_btn = gr.Button("Analyze")
        link_result = gr.HTML(label="Result")
        link_btn.click(fn=predict_link, inputs=link, outputs=link_result, api_name="predict_link")
    
    

demo.launch()