import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from wf_analysis.visualization.theme import Theme


class PlotFactory:
    @staticmethod
    def bar_chart(
        data: pd.DataFrame, x: str, y: str,
        title: str = "", figsize=(10, 6),
    ) -> plt.Figure:
        Theme.set_style()
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=data, x=x, y=y, ax=ax, palette=Theme.CATEGORICAL)
        ax.set_title(title)
        plt.tight_layout()
        return fig

    @staticmethod
    def box_plot(
        data: pd.DataFrame, x: str, y: str,
        hue: str | None = None, title: str = "", figsize=(12, 6),
    ) -> plt.Figure:
        Theme.set_style()
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig

    @staticmethod
    def kde_plot(
        data: pd.DataFrame, x: str,
        hue: str | None = None, title: str = "", figsize=(10, 5),
    ) -> plt.Figure:
        Theme.set_style()
        fig, ax = plt.subplots(figsize=figsize)
        sns.kdeplot(data=data, x=x, hue=hue, ax=ax, fill=True, alpha=0.4)
        ax.set_title(title)
        plt.tight_layout()
        return fig

    @staticmethod
    def heatmap(
        data: pd.DataFrame, annot: bool = True,
        title: str = "", figsize=(10, 8),
    ) -> plt.Figure:
        Theme.set_style()
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data, annot=annot, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        return fig

    @staticmethod
    def pie_chart(
        data: pd.Series, title: str = "", figsize=(8, 8),
    ) -> plt.Figure:
        Theme.set_style()
        fig, ax = plt.subplots(figsize=figsize)
        ax.pie(
            data.values, labels=data.index, autopct="%1.1f%%",
            colors=Theme.CATEGORICAL[: len(data)],
        )
        ax.set_title(title)
        plt.tight_layout()
        return fig
