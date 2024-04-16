import time
import requests
import re
import gradio as gr
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

def format_reviews(amazon_link):
    # Send a GET request to the Amazon link and fetch the HTML content
    response = requests.get(amazon_link)
    if response.status_code != 200:
        print("Failed to fetch page")
        return []

    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all review elements
    review_elements = soup.find_all(class_="a-expander-content reviewText review-text-content a-expander-partial-collapse-content")
    review_ratings = soup.find_all(class_="a-icon a-icon-star a-star-5 review-rating")

    reviews = []
    ratings = []

    # Iterate over each review element and extract the review text
    for review in review_elements:
        # Find the span tag containing the review text
        review_text_tag = review.find('span')
        if review_text_tag:
            # Extract the review text
            review_text = review_text_tag.get_text(strip=True)
            reviews.append(review_text)

    for rating in review_ratings:
        rating_text_tag = rating.find('span')
        if rating_text_tag:
            # Extract the review text
            rating_text = rating_text_tag.get_text(strip=True)
            rating_num = re.search(r'\d+\.\d+', rating_text).group()
            ratings.append(rating_num)

    print(reviews)
    print(ratings)

    return reviews

# def format_reviews(amazon_link):
#     # Set the path to the ChromeDriver executable
#     chrome_driver_path = 'C:\Program Files\chromedriver.exe'

#     # Create a Service object
#     service = Service(chrome_driver_path)

#     # Start a Chrome WebDriver session with the Service object
#     driver = webdriver.Chrome(service=service)
#     driver.get(amazon_link)
#     time.sleep(10)
#     review_elements = driver.find_elements(By.CLASS_NAME, "a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")

#     output = []
# # Iterate over each review element and extract the review text
#     for review_element in review_elements:
#         review_text = review_element.find_element(By.TAG_NAME, "span").text
#         output.append(review_text)
#         # Join reviews into a single string
#     driver.quit()

#     return output

def main():
    # # Define input field for Amazon link
    # input_link = gr.Textbox(lines=1, label="Input Amazon link")

    # # Define output field for reviews
    # output_reviews = gr.Textbox(label="Reviews")

    # # Create Gradio interface
    # gr.Interface(fn=format_reviews, inputs=input_link, outputs=output_reviews, title="Amazon Review Extractor").launch()

    format_reviews('https://www.amazon.com/Wireless-Uiosmuph-Rechargeable-Portable-Computer/dp/B0836GXKKB/ref=pd_gw_rpt_sd_biaws_g_3?_encoding=UTF8&dd=TBNP4Ce6LR_qcvI7mDp3CdNGzeC2M1AGI-JCm90QnS4%2C&ddc_refnmnt=free&pd_rd_i=B0836GXKKB&pd_rd_w=O2ghB&content-id=amzn1.sym.ae3c55fc-7f19-4a1a-be25-ba2c1b81e6de&pf_rd_p=ae3c55fc-7f19-4a1a-be25-ba2c1b81e6de&pf_rd_r=W5F3QBQQ68NN71ZJ3Z78&pd_rd_wg=DeSDz&pd_rd_r=a5fc6ea6-ba84-424f-9b9b-a88bb4e68425&th=1')

if __name__ == "__main__":
    main()