from interval import Interval
from sample import Sample
from utils import UtilHelper
import glob


def main():
    values = [-0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45]
    strings = [str(x) + 'V_sp' for x in values]

    sample = Sample(values)
    for string in strings:
        file_names = glob.glob('../../datasets/3/' + string + '*.dat')
        tmp = list()
        # for file_name in file_names:
        tmp.append(Interval.create_from_data(file_names[0], '../../datasets/3/' + '0.0V_sp812.dat'))
        print(file_names[0])
        sample.append(Interval(min(tmp, key=lambda x: x.begin).begin, max(tmp, key=lambda x: x.end).end))

    for interval in sample.data:
        print(interval)

    sample.draw_sample_plot(title='Interval sample plot')
    coeffs = UtilHelper.calculate_regression(sample)
    print(f"Intervals: beta[0] = {coeffs[0]}, beta[1] = {coeffs[1]}")
    UtilHelper.plot_regression(coeffs, sample, title='Point regression of interval sample')
    inform_set = UtilHelper.build_inform_set(sample, coeffs, plot=True, title='Interval sample inform set')
    UtilHelper.get_corridor(sample, inform_set, title='Interval sample conjoint corridor')

    residuals = UtilHelper.get_residuals(sample, coeffs)
    residuals.draw_sample_plot(title='Residuals sample plot')
    res_coeffs = UtilHelper.calculate_regression(residuals)
    print(f"Residuals: beta[0] = {res_coeffs[0]}, beta[1] = {res_coeffs[1]}")
    UtilHelper.plot_regression(res_coeffs, residuals, title='Point regression of residuals sample')
    res_inform_set = UtilHelper.build_inform_set(residuals, res_coeffs, plot=True, title='Residuals sample inform set')
    corridor = UtilHelper.get_corridor(residuals, res_inform_set, plot=True, title='Residuals sample conjoint corridor')
    UtilHelper.draw_status_diagram(corridor, residuals, title='Residual sample status diagram')

    for interval in residuals.data:
        print(interval)

if __name__ == '__main__':
    main()
