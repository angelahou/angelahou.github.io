"""
A script for scraping data from ESRB website
!!! Make sure these packages and software are installed:
- beautifulsoup (pip install beautifulsoup4)
- selenium (pip install -U selenium)
- webdriver_manager (pip install webdriver_manager --user)
- google chrome (https://www.google.com/chrome/)

Author: Jie Li
Date created: Mar 10, 2021
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
import time
from webdriver_manager.chrome import ChromeDriverManager
import re
driver = webdriver.Chrome(ChromeDriverManager().install())

within_tag = r"<\s*[^>]*>(.*?)<\/.*>"
img_match = r"images/(.*?).svg"

def scrape_single(driver, game_name):
    data = []
    website = f"https://www.esrb.org/search/?searchKeyword={game_name}&platform=Nintendo%20Switch%2CNintendo%203DS%2CWii%20U%2CPlayStation%205%2CPlayStation%204%2CPlayStation%203%2CXbox%20Series%2CXbox%20One%2CXbox%20360%2CStadia%2CPC%2COther&rating=E%2CE10%2B%2CT%2CM%2CAO&descriptor=All%20Content&pg=1&searchType=All"
    driver.get(website)
    for i in range(6):
        content = driver.page_source
        soup = BeautifulSoup(content)
        matches = soup.findAll("div", {"class": "game"})
        if len(matches) == 0:
            # maybe the search result has not yet loaded: wait 0.5s and try again
            time.sleep(0.5)
            continue
        for single_game_div in matches:
            game_name = single_game_div.findAll("h2")[0].findAll("a")[0]
            game_name = re.findall(within_tag, str(game_name))[0]
            platforms = single_game_div.findAll("div",{"class":"platforms"})[0]
            platforms = re.findall(within_tag, str(platforms))[0]
            content_descriptors = single_game_div.findAll("td")[1]
            content_descriptors = re.findall(within_tag, str(content_descriptors))[0]
            rating = single_game_div.findAll("img")[0]
            rating = re.findall(img_match, str(rating))[0]
            data.append({"game_name": game_name, "platforms": platforms, "content_descriptors": content_descriptors, "rating": rating})
        df = pd.DataFrame(data)
        return df


vgsales = pd.read_csv('vgsales.csv')

df = pd.DataFrame()

for name in vgsales['Name'][:10]:
    df = pd.concat([df, scrape_single(driver, name)], ignore_index=True)

df.to_csv('esrb_ratings_sample.csv')