from textblob import TextBlob


blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)

french_res = blob.translate(to="fr")
print(french_res)

# BUG: This file doesn't work properly
# the textblob object has no attribute 'translate'
