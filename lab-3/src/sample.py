from interval import Interval
import matplotlib.pyplot as plt


class Sample:

    def __init__(self, values: list[float]):
        self.values = values
        self.data = list()

    def __len__(self):
        return len(self.data)

    def extend(self, another: 'Sample'):
        self.data.extend(another.data)
        self.__update_borders()

    def __update_borders(self):
        self.lower_min = min(self.data, key=lambda x: x.begin).begin
        self.upper_min = min(self.data, key=lambda x: x.end).end

        self.lower_max = max(self.data, key=lambda x: x.begin).begin
        self.upper_max = max(self.data, key=lambda x: x.end).end

    def append(self, interval: Interval):
        self.data.append(interval)
        self.__update_borders()

    def draw_sample_plot(self, title='', show=True):
        for elem, value in zip(self.data, self.values):
            plt.plot([value, value], [elem.begin, elem.end], c='red')
        if show:
            plt.title(title)
            plt.show()
