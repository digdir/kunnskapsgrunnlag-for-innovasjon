from typing import Generator, Tuple

from .tokenizer import EmbeddingTokenizer

SPLITTERS = [
    ".\n",
    "!\n",
    "?\n",
    ". \n",
    "! \n",
    "? \n",
    ". ",
    "! ",
    "? ",
    ".",
    "?",
    "!",
    "\n",
    " ",
]


# Split the text into paragraphs at every occurrence of two newline characters
# Remove any leading or trailing whitespace from each paragraph
# Remove any paragraphs with length less than 80 characters
def split_data(
    paragraph: str,
    strings_to_split_on: list[str],
    max_length: int,
    tokenizer: EmbeddingTokenizer,
) -> Generator[str, None, None]:
    if tokenizer.len(paragraph) <= max_length:
        return [paragraph]
    if len(strings_to_split_on) == 0:
        return []  # If there are over 512 tokens with no characters to split on, it is safe to say we do not need it
    splitter = strings_to_split_on.pop(0)
    arr = [
        item
        for sublist in [
            split_data(i, strings_to_split_on, max_length, tokenizer)
            for i in paragraph.split(splitter)
        ]
        for item in sublist
    ]

    groups = []
    current_group = []

    for string in arr:
        if tokenizer.len(splitter.join(current_group) + string) > max_length:
            groups.append(current_group)
            current_group = [string]
        else:
            current_group.append(string)

    if current_group:
        groups.append(current_group)

    for g in groups:
        yield splitter.join(g)


def parse_and_split_paragraphs_on_max_length_on_sentence(
    paragraphs: str, tokenizer: EmbeddingTokenizer
) -> Generator[Tuple[int, str], None, None]:
    tokenizer_max_length = tokenizer.tokenizer.model_max_length
    paragraphs = (
        paragraphs.replace("\x03", "\n\n").replace("....", "").replace(". . ", "")
    )
    pages = paragraphs.split("\x0c")
    for page_number, page in enumerate(pages, start=1):
        for paragraph in page.split("\n\n"):
            # if len(paragraph) < 80:
            #     continue
            if tokenizer.len(paragraph) <= tokenizer_max_length:
                yield page_number, paragraph
                continue
            for i in split_data(
                paragraph,
                SPLITTERS,
                tokenizer_max_length,
                tokenizer,
            ):
                yield page_number, i
