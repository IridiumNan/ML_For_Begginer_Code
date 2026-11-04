from html.parser import HTMLParser


class ParagraphExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_p = False
        self.paragraphs = []
        self.current_text = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "p":
            self.in_p = True
            self.current_text = []

    def handle_endtag(self, tag):
        if tag.lower() == "p":
            self.in_p = False
            text = "".join(self.current_text).strip()
            if text:
                self.paragraphs.append(text)

    def handle_data(self, data):
        if self.in_p:
            self.current_text.append(data)


def extract_paragraphs_native(file_path):
    """Extracts text from all <p> tags using standard library."""
    parser = ParagraphExtractor()
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            parser.feed(file.read())
        return parser.paragraphs
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []


# Example usage:
if __name__ == "__main__":
    html_file = "./1342-h.htm"
    extracted_texts = extract_paragraphs_native(html_file)

    with open("raw_p.txt", "w") as f:
        for i, text in enumerate(extracted_texts):
            # skip the useless index
            if i <= 2 or i == 2139:
                continue
            print(f"  {text}", file=f)
