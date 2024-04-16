import gradio as gr
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

def format_reviews(amazon_link):
    # Set the path to the ChromeDriver executable
    chrome_driver_path = 'C:\Program Files\chromedriver.exe'

    # Create a Service object
    service = Service(chrome_driver_path)

    # Start a Chrome WebDriver session with the Service object
    driver = webdriver.Chrome(service=service)
    driver.get(amazon_link)
    time.sleep(10)
    review_elements = driver.find_elements(By.CLASS_NAME, "a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")

    output = []
# Iterate over each review element and extract the review text
    for review_element in review_elements:
        review_text = review_element.find_element(By.TAG_NAME, "span").text
        output.append(review_text)
        # Join reviews into a single string
    driver.quit()

    return output

def main():
    # # Define input field for Amazon link
    # input_link = gr.Textbox(lines=1, label="Input Amazon link")

    # # Define output field for reviews
    # output_reviews = gr.Textbox(label="Reviews")

    # # Create Gradio interface
    # gr.Interface(fn=format_reviews, inputs=input_link, outputs=output_reviews, title="Amazon Review Extractor").launch()

    print(format_reviews('https://www.amazon.com/Wireless-Uiosmuph-Rechargeable-Portable-Computer/dp/B0836GXKKB/ref=pd_gw_rpt_sd_biaws_g_3?_encoding=UTF8&dd=TBNP4Ce6LR_qcvI7mDp3CdNGzeC2M1AGI-JCm90QnS4%2C&ddc_refnmnt=free&pd_rd_i=B0836GXKKB&pd_rd_w=O2ghB&content-id=amzn1.sym.ae3c55fc-7f19-4a1a-be25-ba2c1b81e6de&pf_rd_p=ae3c55fc-7f19-4a1a-be25-ba2c1b81e6de&pf_rd_r=W5F3QBQQ68NN71ZJ3Z78&pd_rd_wg=DeSDz&pd_rd_r=a5fc6ea6-ba84-424f-9b9b-a88bb4e68425&th=1'))

if __name__ == "__main__":
    main()