import argparse
import os
import re

import requests
from bs4 import BeautifulSoup


######################
# Text Scraping Functions
######################
def scrape_links(num_articles):
    """Scrape links and save off text to csv files"""

    # Identify location to save off files
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "shared",
        "data",
        "wikipedia",
    )

    # Scrape the links
    for _ in range(num_articles):
        # Get the wikipedia page
        page = requests.get("https://en.wikipedia.org/wiki/Special:Random")
        soup = BeautifulSoup(page.content, "html.parser")
        # Get the title of the page
        title = soup.find(id="firstHeading").get_text()
        # Get the text of the page
        paragraphs = soup.find_all("p")
        text = ""
        for paragraph in paragraphs:
            text += paragraph.get_text()
        # Remove references
        text = re.sub(r"\[[^]]*\]", "", text)

        # Save off the text to a csv
        with open(os.path.join(data_path, title + ".csv"), "w") as f:
            f.write(text)
    return 0


######################
# Clean/Preprocess Text Functions
######################
def clean_text(text):
    pass


def tokenize_text(text, tokenizer):
    pass


######################
# Argument Parsing
######################
def argument_parser():
    """Parse the command-line arguments"""
    # Define the command-line arguments
    parser = argparse.ArgumentParser(description="scrape_wikipedia.py")

    # Add arguments
    parser.add_argument(
        "--num", "-n", type=int, help="The number of articles to scrape from wikipedia"
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


######################
# Main
######################
def main():
    """Main function"""
    # Parse arguments
    args = argument_parser()

    # Scrape Wikipedia page for try:
    scrape_links(args.num)

    return 0


if __name__ == "__main__":
    main()
