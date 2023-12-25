import argparse
import os
import re
from enum import Enum

import nltk
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup


class ExampleTypes(Enum):
    """Types of examples to generate for word2vec"""

    CBOW = "cbow"
    SKIPGRAM = "skipgram"


SKIPGRAM_DATA_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "shared",
    "data",
    "skipgram",
)


CBOW_DATA_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "shared",
    "data",
    "cbow",
)

EXAMPLE_TYPE_TO_FOLDER = {
    ExampleTypes.CBOW.value: CBOW_DATA_FOLDER,
    ExampleTypes.SKIPGRAM.value: SKIPGRAM_DATA_FOLDER,
}


######################
# Text Preprocessing Functions
######################
def text_to_cbow_examples(text: str, radius: int) -> list[tuple[str, str]]:
    """Convert text to CBOW examples"""
    examples = []
    words = text.split()
    for i, word in enumerate(words):
        # Skip if word does not have enough context before or after
        if i < radius or i >= len(words) - radius:
            continue
        # Get context words
        center_word = word
        context_words = words[i - radius : i] + words[i + 1 : i + radius + 1]
        # Create examples
        examples.append((context_words, center_word))
    return examples


def text_to_skipgram_examples(text: str, radius: int) -> list[tuple[str, str]]:
    """Convert text to skipgram examples"""
    examples = []
    words = text.split()
    for i, word in enumerate(words):
        # Skip if word does not have enough context before or after
        if i < radius or i >= len(words) - radius:
            continue
        # Get context words
        center_word = word
        context_words = words[i - radius : i] + words[i + 1 : i + radius + 1]
        # Create examples
        for context_word in context_words:
            examples.append((center_word, context_word))
    return examples


EXAMPLE_TYPE_TO_FUNCTION = {
    ExampleTypes.CBOW.value: text_to_cbow_examples,
    ExampleTypes.SKIPGRAM.value: text_to_skipgram_examples,
}


# Define a function to get file path
def get_output_file_path(counter, example_type):
    output_folder = EXAMPLE_TYPE_TO_FOLDER[example_type]
    return os.path.join(output_folder, f"{counter:03}.txt")


def create_embedding_dataset(data_folder: str, example_type: str, window_size: int):
    """Create embedding model dataset"""

    data_files = os.listdir(data_folder)
    example_counter = 0
    file_counter = 1
    max_examples_per_file = 2048
    radius = window_size // 2

    # Open the first output file
    output_file_path = get_output_file_path(file_counter, example_type)
    output_file = open(output_file_path, "w")

    for file_name in data_files:
        # Read and clean text
        with open(os.path.join(data_folder, file_name), "r") as file:
            try:
                text = file.read()
            except UnicodeDecodeError:
                continue

        # Convert text to examples and write to file
        examples = EXAMPLE_TYPE_TO_FUNCTION[example_type](text, radius)
        for example in examples:
            output_file.write(f"{example}\n")
            example_counter += 1

            # Check if the current file has reached its limit
            if example_counter >= max_examples_per_file:
                output_file.close()  # Close current file
                file_counter += 1  # Increment file counter
                output_file_path = get_output_file_path(file_counter, example_type)
                output_file = open(output_file_path, "w")  # Open new file
                example_counter = 0  # Reset example counter

    # Close the last file if it's still open
    if not output_file.closed:
        output_file.close()


######################
# Argument Parsing
######################
def argument_parser():
    """Parse the command-line arguments"""
    # Define the command-line arguments
    parser = argparse.ArgumentParser(description="scrape_wikipedia.py")

    # Add arguments
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        help="Folder containing the cleaned text files",
    )
    parser.add_argument(
        "--example_type",
        "-e",
        type=str,
        help="Type of examples to generate. Must be `cbow` or `skipgram`",
    )
    parser.add_argument(
        "--window_size",
        "-w",
        type=int,
        help="Window size for `cbow` or `skipgram` to relate center word to context word(s)",
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
    assert args.example_type in EXAMPLE_TYPE_TO_FUNCTION.keys()
    assert args.window_size > 0
    assert args.window_size % 2 == 0
    assert args.window_size < 20

    # Preprocess and save the dataset
    create_embedding_dataset(args.data_path, args.example_type, args.window_size)

    return 0


if __name__ == "__main__":
    main()
