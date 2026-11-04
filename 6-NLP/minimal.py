from textblob import TextBlob
from textblob.np_extractors import ConllExtractor

extractor = ConllExtractor()

prompt = "> "

while True:
    user_input = input(prompt)

    if user_input == "exit":
        print("bye")
        exit(0)

    user_input_blob = TextBlob(user_input, np_extractor=extractor)

    polarity: float = user_input_blob.polarity

    # default response for [0.5, 1]
    response = "Wow, that sounds great"

    if polarity <= -0.5:
        response = "Oh dear, that sounds bad."
    elif polarity <= 0:
        response = "Hmm, that's not great."
    elif polarity <= 0.5:
        response = "Well, that sounds positive"

    print(response)
