import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from wf_analysis.visualization.theme import Theme


class DashboardBuilder:
    @staticmethod
    def build_grid(
        plots: list[plt.Figure],
        layout: tuple[int, int] | None = None,
    ) -> plt.Figure:
        Theme.set_style()
        n = len(plots)
        if n == 0:
            return plt.figure(figsize=(6, 4))
        if layout:
            rows, cols = layout
        else:
            cols = min(3, n)
            rows = (n + cols - 1) // cols

        fig = plt.figure(figsize=(cols * 6, rows * 5))
        gs = GridSpec(rows, cols, figure=fig)

        for i, p in enumerate(plots):
            if i >= rows * cols:
                break
            ax = fig.add_subplot(gs[i // cols, i % cols])
            for child in p.get_axes():
                for line in child.get_lines():
                    ax.plot(line.get_xdata(), line.get_ydata())
            ax.set_title(p.axes[0].get_title() if p.axes else "")
            plt.close(p)

        plt.tight_layout()
        return fig
