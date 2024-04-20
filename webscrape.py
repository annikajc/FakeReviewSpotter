import time
from selenium import webdriver
from bs4 import BeautifulSoup
import re

def format_reviews(amazon_link):
    # Set up Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # To run Chrome in headless mode
    driver = webdriver.Chrome(options=options)

    amazon_link = amazon_link.replace('dp', 'product-reviews')

    reviews = []

    for i in range(2, 12):
        new_amazon_link = amazon_link + f"/ref=cm_cr_arp_d_paging_btm_next_{i}?&pageNumber={i}"
        driver.get(new_amazon_link)

        # Scroll to the bottom of the page to load more reviews
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Parse the HTML content using Beautiful Soup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Find all review elements
        review_elements = soup.find_all(class_='a-section celwidget')

        if (len(review_elements) == 0):
            break

        # Iterate over each review element and extract the review text
        for review in review_elements:
            
            review_text_tag = review.find('span', {'data-hook': 'review-body'})
            if review_text_tag:
                review_text = review_text_tag.get_text(strip=True)
            
            rating_text_tag = review.find(class_='a-icon-alt')
            if rating_text_tag:
                rating_text = rating_text_tag.get_text(strip=True)
                if re.search(r'\d+\.\d+', rating_text) is not None:
                    rating_num = re.search(r'\d+\.\d+', rating_text).group()
            
            tuple = (review_text, float(rating_num))

            reviews.append(tuple)

    driver.quit()

    return reviews
