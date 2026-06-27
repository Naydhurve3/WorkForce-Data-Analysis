import matplotlib.pyplot as plt
import seaborn as sns


class Theme:
    PRIMARY = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
    DIVERGING = ["#2E86AB", "#F18F01", "#C73E1D"]
    CATEGORICAL = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
        "#6AB547", "#E58C6A", "#8B5CF6", "#06B6D4", "#84CC16",
    ]

    @classmethod
    def set_style(cls):
        sns.set_theme(style="whitegrid", palette=cls.CATEGORICAL)
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        })

    @classmethod
    def get_palette(cls, name: str = "categorical") -> list:
        if name == "primary":
            return cls.PRIMARY
        elif name == "diverging":
            return cls.DIVERGING
        return cls.CATEGORICAL
