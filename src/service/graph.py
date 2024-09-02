import matplotlib.pyplot as plt
import numpy as np

# Data from the dictionary
techniques = ['bm25_search', 'tfifd', 'embedding', 'fuzzy']
scores = [6.1888895, 6.4888895, 6.0888895, 6.6888895]
rerank_scores = [7.609801042906972, 7.509801042906972, 7.409801042906972, 7.009801042906972]


def create_side_by_side_bar_graph(techniques, rerank_scores, scores):
    # Number of techniques
    n = len(techniques)

    # Create a range of values for x-axis
    ind = np.arange(n)

    # Bar width
    width = 0.35

    # Plotting the data
    fig, ax = plt.subplots()

    # Bars for rerankScore with a different color
    rerank_bars = ax.bar(ind - width / 2, rerank_scores, width, label='Rerank Score', color='skyblue')

    # Bars for score with a different color
    score_bars = ax.bar(ind + width / 2, scores, width, label='Score', color='lightgreen')

    # Adding labels, title, and custom x-axis tick labels
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Rerank Score and Score across Search Techniques')
    ax.set_xticks(ind)
    ax.set_xticklabels(techniques)
    ax.legend()

    # Function to add labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Adding labels
    add_labels(rerank_bars)
    add_labels(score_bars)

    # Display the plot
    plt.show()


def create_stacked_bar_graph(techniques, rerank_scores, scores):
    # Number of techniques
    n = len(techniques)

    # Create a range of values for x-axis
    ind = np.arange(n)

    # Plotting the data
    fig, ax = plt.subplots()

    # Stacked bars
    ax.bar(ind, rerank_scores, label='Rerank Score', color='skyblue')
    ax.bar(ind, scores, bottom=rerank_scores, label='Score', color='lightgreen')

    # Adding labels, title, and custom x-axis tick labels
    ax.set_ylabel('Scores')
    ax.set_title('Stacked Comparison of Rerank Score and Score across Search Techniques')
    ax.set_xticks(ind)
    ax.set_xticklabels(techniques)
    ax.legend()

    # Display the plot
    plt.show()


def create_horizontal_bar_graph(techniques, rerank_scores, scores):
    # Number of techniques
    n = len(techniques)

    # Create a range of values for y-axis
    ind = np.arange(n)

    # Bar width
    width = 0.35

    # Plotting the data
    fig, ax = plt.subplots()

    # Horizontal bars for rerankScore
    rerank_bars = ax.barh(ind - width / 2, rerank_scores, width, label='Rerank Score', color='skyblue')

    # Horizontal bars for score
    score_bars = ax.barh(ind + width / 2, scores, width, label='Score', color='lightgreen')

    # Adding labels, title, and custom y-axis tick labels
    ax.set_xlabel('Scores')
    ax.set_title('Horizontal Bar Graph: Comparison of Rerank Score and Score across Search Techniques')
    ax.set_yticks(ind)
    ax.set_yticklabels(techniques)
    ax.legend()

    # Function to add labels on bars
    def add_labels(bars):
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.2f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center')

    # Adding labels
    add_labels(rerank_bars)
    add_labels(score_bars)

    # Display the plot
    plt.show()


def create_pie_chart_rerank_scores(techniques, rerank_scores):
    # Plotting the data
    fig, ax = plt.subplots()

    # Pie chart for rerank scores
    ax.pie(rerank_scores, labels=techniques, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)

    # Adding title
    ax.set_title('Distribution of Rerank Scores Across Search Techniques')

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')

    # Display the plot
    plt.show()


create_pie_chart_rerank_scores(techniques, rerank_scores)
create_horizontal_bar_graph(techniques, rerank_scores, scores)
create_stacked_bar_graph(techniques, rerank_scores, scores)
create_side_by_side_bar_graph(techniques, rerank_scores, scores)
