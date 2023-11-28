
import plotly.graph_objs as go

class PlotMaker:

    def __init__(self, series,go_trace, title, xaxis_title, yaxis_title, color=None, mode=None):
        self.series = series
        self.go_trace = go_trace
        self.title = title
        self.xaxis_title = xaxis_title
        self.yaxis_title = yaxis_title
        self.color = color
        self.mode = mode
        self.plot = self.create_plot()

    def create_plot(self):
        plot = go.Figure(layout=go.Layout(
            title=self.title,
            xaxis=dict(title=self.xaxis_title),
            yaxis=dict(title=self.yaxis_title), ), )

        plot.add_trace(self.go_trace(y=self.series, x=self.series.index, marker=dict(color=self.color)))
        return plot
