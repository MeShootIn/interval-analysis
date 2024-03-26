import scipy.optimize as opt
import scipy.spatial as spa
from matplotlib.patches import Polygon

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
    def plot_regression(coeffs: tuple[float, float], sample: Sample, title: str, offset: float = 0.25):
        x = [sample.values[0] - offset]
        x.extend(sample.values)
        x.append(sample.values[-1] + offset)

        y = [coeffs[0] + coeffs[1] * x_i for x_i in x]
        plt.plot(x, y)
        sample.draw_sample_plot('', show=False)
        plt.title(title)
        plt.show()

    @staticmethod
    def build_inform_set(sample: Sample, beta_opt: tuple[float, float], plot: bool = False):
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
            plt.title('Inform set')
            plt.scatter(beta_opt[0], beta_opt[1])
            plt.xlabel('beta_0')
            plt.ylabel('beta_1')
            plt.show()
        return hs

    @staticmethod
    def plot_corridor(sample, hs, offset=0.25):
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

        raw_poly = [(x, y) for x, y in zip(x, y_max)]
        raw_poly.extend(reversed([(x, y) for x, y in zip(x, y_min)]))

        fig, ax = plt.subplots(1)
        polygon = Polygon(raw_poly, alpha=0.5)
        ax.add_patch(polygon)
        sample.draw_sample_plot('', show=False)
        plt.title('Corridor of conjoint values')
        plt.show()
