import gradio as gr
from predict import is_fake

def predict_review(review):
    """
    Print output for review analysis
    """
    if is_fake(review):
        return "Review is fake."
    else:
        return "Review is real."


def predict_link(link):
    """
    Output for Amazon link analysis
    """
    # FILLER RETURN STATEMENT
    return "INSERT RESULT HERE"


with gr.Blocks() as demo:
    gr.Markdown("# Analyze fake reviews here!")
    gr.Markdown("Assess if a review is fake, or assess the percentage of fake reviews for an Amazon product.")

    with gr.Tab("Analyze Review"):
        review = gr.Textbox(label="Insert review")
        review_result = gr.Text(label="Result")
        review_btn = gr.Button("Analyze")
        review_btn.click(fn=predict_review, inputs=review, outputs=review_result, api_name="predict_review")

    with gr.Tab("Analyze Link"):
        link = gr.Textbox(label="Insert Amazon link")
        link_result = gr.Text(label="Result")
        link_btn = gr.Button("Analyze")
        link_btn.click(fn=predict_link, inputs=link, outputs=link_result, api_name="predict_link")
    
    

demo.launch()