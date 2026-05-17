"""Creates a word cloud from a list of tokens."""

OUTPUT_DIR = "output"


def create_word_cloud(tokens: list[str]) -> None:
    """Creates a word cloud from a list of tokens."""
    import matplotlib
    from wordcloud import WordCloud

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Join the tokens into a single string
    text = " ".join(tokens)

    # Create a word cloud object
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Save the generated word cloud to a file instead of showing it
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/wordcloud.png")
    plt.close()
