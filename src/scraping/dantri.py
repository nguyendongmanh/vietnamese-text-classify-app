import requests
from bs4 import BeautifulSoup


def get_soup(url: str, headers: dict = None, timeout: int = 60):
    response = requests.get(url, timeout=timeout, headers=headers)
    return BeautifulSoup(response.text, "lxml")


def get_content(url: str, headers: dict = None, timeout: int = 60):
    content = None
    div_tag = None

    soup = get_soup(url, headers, timeout)
    article = soup.find("article")
    article_type = article["class"]

    title = soup.find("h1").text
    if "d-magazine" in article_type:
        div_tag = soup.find("div", class_="e-magazine__body dnews__body")
    elif "singular-container" in article_type:
        div_tag = soup.find("div", class_="singular-content")

    if div_tag:
        content = div_tag.text

    return content, title
