import time
import requests
import re
import gradio as gr
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

# def extract_text(reviews, ratings, review_elements, review_ratings):
#     # Iterate over each review element and extract the review text
#     for review in review_elements:
#         # Find the span tag containing the review text
#         review_text_tag = review.find('span')
#         if review_text_tag:
#             # Extract the review text
#             review_text = review_text_tag.get_text(strip=True)
#             reviews.append(review_text)

#     for rating in review_ratings:
#         rating_text_tag = rating.find('span')
#         if rating_text_tag:
#             # Extract the review text
#             rating_text = rating_text_tag.get_text(strip=True)
#             #rating_text = rating.get_text(strip=True)
#             rating_num = re.search(r'\d+\.\d+', rating_text).group()
#             ratings.append(rating_num)

def format_reviews(amazon_link):
    # driver = webdriver.Chrome("./chromedriver.exe")
    # driver.get(amazon_link)

    amazon_link.replace("dp", "product-reviews")

    # Send a GET request to the Amazon link and fetch the HTML content
    # response = requests.get(amazon_link)
    # if response.status_code != 200:
    #     print("Failed to fetch page")
    #     return []

    # # Parse the HTML content using Beautiful Soup
    # soup = BeautifulSoup(response.text, 'html.parser')

    # # Find all review elements
    # review_elements = soup.find_all(class_="a-expander-content reviewText review-text-content a-expander-partial-collapse-content")
    # review_ratings = soup.find_all(class_="a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold")

    reviews = []
    ratings = []

    # # Iterate over each review element and extract the review text
    # for review in review_elements:
    #     # Find the span tag containing the review text
    #     review_text_tag = review.find('span')
    #     if review_text_tag:
    #         # Extract the review text
    #         review_text = review_text_tag.get_text(strip=True)
    #         reviews.append(review_text)

    # for rating in review_ratings:
    #     rating_text_tag = rating.find('span')
    #     if rating_text_tag:
    #         # Extract the review text
    #         rating_text = rating_text_tag.get_text(strip=True)
    #         #rating_text = rating.get_text(strip=True)
    #         rating_num = re.search(r'\d+\.\d+', rating_text).group()
    #         ratings.append(rating_num)

    for i in range(500, 501):
        new_amazon_link = amazon_link + f"/ref=cm_cr_arp_d_paging_btm_next_2?&pageNumber={i}"
        print(new_amazon_link + "\n")
        response = requests.get(new_amazon_link)
        time.sleep(2)

        # Parse the HTML content using Beautiful Soup
        soup = BeautifulSoup(response.text, 'html.parser')

        target_div = soup.find("div", {"id": "filter-info-section"})

        ahhhhh = []
        # Extract text if the div is found
        if target_div:
            print(target_div.text)
        else:
            print("Target div not found")

        # Find all review elements
        review_elements = soup.find_all(class_="a-expander-content reviewText review-text-content a-expander-partial-collapse-content")
        review_ratings = soup.find_all(class_="a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold")

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
                #rating_text = rating.get_text(strip=True)
                rating_num = re.search(r'\d+\.\d+', rating_text).group()
                ratings.append(rating_num)


    # print(reviews)
    # print(ratings)

    return reviews

def main():
    # #Define input field for Amazon link
    # input_link = gr.Textbox(lines=1, label="Input Amazon link")

    # #Define output field for reviews
    # output_reviews = gr.Textbox(label="Reviews")

    # #Create Gradio interface
    # gr.Interface(fn=format_reviews, inputs=input_link, outputs=output_reviews, title="Amazon Review Extractor").launch()

    format_reviews('https://www.amazon.com/Wireless-Uiosmuph-Rechargeable-Portable-Computer/dp/B0836GXKKB/ref=pd_gw_rpt_sd_biaws_g_3?_encoding=UTF8&dd=TBNP4Ce6LR_qcvI7mDp3CdNGzeC2M1AGI-JCm90QnS4%2C&ddc_refnmnt=free&pd_rd_i=B0836GXKKB&pd_rd_w=O2ghB&content-id=amzn1.sym.ae3c55fc-7f19-4a1a-be25-ba2c1b81e6de&pf_rd_p=ae3c55fc-7f19-4a1a-be25-ba2c1b81e6de&pf_rd_r=W5F3QBQQ68NN71ZJ3Z78&pd_rd_wg=DeSDz&pd_rd_r=a5fc6ea6-ba84-424f-9b9b-a88bb4e68425&th=1')

if __name__ == "__main__":
    main()