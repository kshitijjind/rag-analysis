import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from logger import logger
import os


def create_side_by_side_bar_graph(techniques, rerank_scores, scores, ax):
    n = len(techniques)
    ind = np.arange(n)
    width = 0.35

    rerank_bars = ax.bar(ind - width / 2, rerank_scores, width, label='Rerank Score', color='skyblue')
    score_bars = ax.bar(ind + width / 2, scores, width, label='Score', color='lightgreen')

    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Rerank Score and Score across Search Techniques')
    ax.set_xticks(ind)
    ax.set_xticklabels(techniques)
    ax.legend()

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(rerank_bars)
    add_labels(score_bars)


def create_stacked_bar_graph(techniques, rerank_scores, scores, ax):
    n = len(techniques)
    ind = np.arange(n)

    ax.bar(ind, rerank_scores, label='Rerank Score', color='skyblue')
    ax.bar(ind, scores, bottom=rerank_scores, label='Score', color='lightgreen')

    ax.set_ylabel('Scores')
    ax.set_title('Stacked Comparison of Rerank Score and Score across Search Techniques')
    ax.set_xticks(ind)
    ax.set_xticklabels(techniques)
    ax.legend()


def create_horizontal_bar_graph(techniques, rerank_scores, scores, ax):
    n = len(techniques)
    ind = np.arange(n)
    width = 0.35

    rerank_bars = ax.barh(ind - width / 2, rerank_scores, width, label='Rerank Score', color='skyblue')
    score_bars = ax.barh(ind + width / 2, scores, width, label='Score', color='lightgreen')

    ax.set_xlabel('Scores')
    ax.set_title('Horizontal Bar Graph: Comparison of Rerank Score and Score across Search Techniques')
    ax.set_yticks(ind)
    ax.set_yticklabels(techniques)
    ax.legend()

    def add_labels(bars):
        for bar in bars:
            width = bar.get_width()
            ax.annotate(f'{width:.2f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center')

    add_labels(rerank_bars)
    add_labels(score_bars)


def create_pie_chart_rerank_scores(techniques, rerank_scores, ax):
    ax.pie(rerank_scores, labels=techniques, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    ax.set_title('Distribution of Rerank Scores Across Search Techniques')
    ax.axis('equal')


def create_all_graphs_to_pdf(techniques, rerank_scores, scores, correlation_id):
    # Define the path for the PDF
    pdf_dir = 'reports'
    pdf_path = os.path.join(pdf_dir, f'comparison_report_{correlation_id}.pdf')

    # Create the 'reports' directory if it does not exist
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)

    # Create a PDF file to save all graphs
    with PdfPages(pdf_path) as pdf:
        # Side-by-side bar graph
        fig, ax = plt.subplots()
        create_side_by_side_bar_graph(techniques, rerank_scores, scores, ax)
        pdf.savefig(fig)
        plt.close(fig)

        # Stacked bar graph
        fig, ax = plt.subplots()
        create_stacked_bar_graph(techniques, rerank_scores, scores, ax)
        pdf.savefig(fig)
        plt.close(fig)

        # Horizontal bar graph
        fig, ax = plt.subplots()
        create_horizontal_bar_graph(techniques, rerank_scores, scores, ax)
        pdf.savefig(fig)
        plt.close(fig)

        # Pie chart for rerank scores
        fig, ax = plt.subplots()
        create_pie_chart_rerank_scores(techniques, rerank_scores, ax)
        pdf.savefig(fig)
        plt.close(fig)

    logger.info(f"PDF report generated with correlation_id: {correlation_id}")
