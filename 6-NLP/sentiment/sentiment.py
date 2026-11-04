from textblob.blob import TextBlob
import subprocess
import sys
import os

raw_content_file = "./raw_p.txt"
csv_file = "./sentiment.csv"


def demo():

    quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

    quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

    sentiment1 = TextBlob(quote1).sentiment
    sentiment2 = TextBlob(quote2).sentiment

    print(f"sentiment1: {sentiment1.polarity}")
    print(f"sentiment2: {sentiment2}")


def process_pride_and_prejudice():

    with open(raw_content_file, "r") as f:
        content_lines = f.readlines()

    # wirte the output into a csv file

    # header
    # id,polarity,subjectivity
    # 1,xxx,xxx

    csv_output = "id,polarity,subjectivity\n"

    with open(csv_file, "w") as out:
        print(csv_output, file=out)
        id = 0
        for line in content_lines:
            sentiment = TextBlob(line).sentiment
            id += 1
            print(f"{id},{sentiment.polarity},{sentiment.subjectivity}", file=out)


def parse_pride_and_prejudice():
    """
    This function use pandas to parse the output of process
    """
    import pandas as pd

    df = pd.read_csv(csv_file)

    print(df.describe())

    negative_df = df[(df["polarity"] < 0)]

    positive_df = df[(df["polarity"] > 0)]

    print("the length of negtive_df: ", len(negative_df))
    print("the length of positive_df: ", len(positive_df))

    print("\nall avarge: \n", df.mean(axis=0))
    print("\nnegative avarge: \n", negative_df.mean(axis=0))
    print("\npositive avarge: \n", positive_df.mean(axis=0))


def main():
    if not os.path.exists(csv_file):
        print(
            "csv file not found, try to read then process from raw content", flush=True
        )
        if not os.path.exists(raw_content_file):
            print("raw content not found, try to extract")
            print(f"running command:  {sys.executable} extract_p_tag.py", flush=True)

            # os.execvp(sys.executable, [sys.executable, "extract_p_tag.py"])
            # use subprocess to create new process instead of replace current process
            subprocess.run([sys.executable, "extract_p_tag.py"], check=True)

        process_pride_and_prejudice()

    parse_pride_and_prejudice()


if __name__ == "__main__":
    main()
