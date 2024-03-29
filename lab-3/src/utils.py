import scipy.optimize as opt
import scipy.spatial as spa
from matplotlib.patches import Polygon
from interval import Interval
from sample import Sample
import matplotlib.pyplot as plt
import numpy as np


class UtilHelper:
    @staticmethod
    def calculate_regression(sample: Sample, param_count: int = 2):
        c = [1 for _ in range(len(sample))]
        c.extend([0 for _ in range(param_count)])

        A_ub = list()
        for i in range(len(sample)):
            tmp_constraint = [0 for _ in range(len(sample))]
            tmp_constraint.extend([-1, -sample.values[i]])  # bruuuuh
            tmp_constraint[i] = -sample.data[i].rad()
            A_ub.append(tmp_constraint)

            tmp_constraint = [0 for _ in range(len(sample))]
            tmp_constraint.extend([1, sample.values[i]])  # bruuuuh
            tmp_constraint[i] = -sample.data[i].rad()
            A_ub.append(tmp_constraint)

        b_ub = list()
        for i in range(len(sample)):
            b_ub.append(-sample.data[i].mid())
            b_ub.append(sample.data[i].mid())

        bounds = [(0, None) for _ in range(len(sample))]
        # noinspection PyTypeChecker
        bounds.extend([(None, None) for _ in range(param_count)])

        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

        return res.x[-2], res.x[-1]

    @staticmethod
    def plot_regression(coeffs: tuple[float, float], sample: Sample, title: str = '', offset: float = 0.25):
        x = [sample.values[0] - offset]
        x.extend(sample.values)
        x.append(sample.values[-1] + offset)

        y = [coeffs[0] + coeffs[1] * x_i for x_i in x]
        plt.plot(x, y)
        sample.draw_sample_plot(show=False)
        plt.title(title)
        plt.show()

    @staticmethod
    def build_inform_set(sample: Sample, beta_opt: tuple[float, float], plot: bool = False, title: str = ''):
        tmp = list()
        for i in range(len(sample)):
            tmp.append(np.array([1, sample.values[i], -sample.data[i].end]))
            tmp.append(np.array([-1, -sample.values[i], sample.data[i].begin]))
        halfspaces = np.array(tmp)
        feasible_point = np.array(beta_opt)

        hs = spa.HalfspaceIntersection(halfspaces, feasible_point)

        x, y = list(), list()
        hull = spa.ConvexHull(hs.intersections)
        for vertex in hull.vertices:
            x.append(hs.intersections[vertex][0])
            y.append(hs.intersections[vertex][1])
        x.append(x[0])
        y.append(y[0])
        if plot:
            plt.plot(x, y)
            plt.scatter(beta_opt[0], beta_opt[1])
            plt.title(title)
            plt.show()
        return hs

    @staticmethod
    def get_corridor(sample, hs, offset=0.25, plot=True, title: str = ''):
        raw_lines = list()
        for a in hs.intersections:
            raw_lines.append((a[0], a[1]))

        coeff_list = list()
        for pair in hs.intersections:
            coeff_list.append((pair[0], pair[1]))

        x = [sample.values[0] - offset]
        x.extend(sample.values)
        x.append(sample.values[-1] + offset)

        y = list()
        for coeffs in coeff_list:
            y.extend([coeffs[0] + coeffs[1] * x_i for x_i in x])

        i = 0
        y_slices = list()
        while i < len(y):
            y_slices.append(y[i:i + len(x)])
            i += len(x)

        y_min = list()
        y_max = list()
        for i in range(len(x)):
            tmp = list()
            for y_slice in y_slices:
                tmp.append(y_slice[i])
            y_max.append(max(tmp))
            y_min.append(min(tmp))

        corridor = Sample(sample.values)
        corridor.data = [Interval(y_mi, y_ma) for y_mi, y_ma in zip(y_min[1:len(y_min) - 1],
                                                                    y_max[1:len(y_max) - 1])]

        if plot:
            raw_poly = [(x, y) for x, y in zip(x, y_max)]
            raw_poly.extend(reversed([(x, y) for x, y in zip(x, y_min)]))
            fig, ax = plt.subplots(1)
            UtilHelper.__plot_polygon(raw_poly, ax)
            sample.draw_sample_plot(show=False)
            plt.title(title)
            plt.show()

        return corridor

    @staticmethod
    def get_residuals(sample, coeffs):
        res_sample = Sample(sample.values)
        for interval, value in zip(sample.data, sample.values):
            res_sample.append(Interval(interval.begin - (coeffs[0] + coeffs[1] * value),
                                       interval.end - (coeffs[0] + coeffs[1] * value)))
        return res_sample

    @staticmethod
    def __l(sample_interval: Interval, model_interval: Interval):
        return model_interval.rad() / sample_interval.rad()

    @staticmethod
    def __r(sample_interval: Interval, model_interval: Interval):
        return (sample_interval.mid() - model_interval.mid()) / sample_interval.rad()

    @staticmethod
    def __plot_polygon(vertices, ax, color=None, edge_color='black'):
        if color is None:
            polygon = Polygon(vertices, alpha=0.5)
        else:
            polygon = Polygon(vertices, alpha=0.5, fc=color, ec=edge_color)
        ax.add_patch(polygon)

    @staticmethod
    def draw_status_diagram(sample, residuals, title=''):
        _, ax = plt.subplots(1)
        UtilHelper.__plot_polygon([[0, -1], [2, -3], [0, -3]], ax, 'red')
        UtilHelper.__plot_polygon([[0, 1], [2, 3], [0, 3]], ax, 'red')
        UtilHelper.__plot_polygon([[0, -1], [1, 0], [0, 1]], ax, 'green')
        UtilHelper.__plot_polygon([[0, -1], [1, 0], [0, 1], [2, 3], [2, -3]], ax, 'yellow')
        plt.xlim([0, 2])
        plt.ylim([-3, 3])
        plt.plot([1, 1], [-3, 3], '--', c='black')

        x, y = list(), list()
        for interval, res in zip(sample.data, residuals.data):
            x.append(UtilHelper.__l(res, interval))
            y.append(UtilHelper.__r(res, interval))
        ax.scatter(x, y, ec='black')

        plt.title(title)
        plt.show()
