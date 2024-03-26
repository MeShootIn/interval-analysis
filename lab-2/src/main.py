from interval import Interval
from sample import Sample
from utils import UtilHelper


def main():
    values = [-0.5, -0.25, 0.25, 0.5]
    strings = ['-0_5V', '-0_25V', '+0_25V', '+0_5V']
    number = 4

    sample = Sample(values)
    for string in strings:
        sample.append(Interval.create_from_data('../../datasets/' + string + '/' + string + '_' + str(number) + '.txt',
                                                '../../datasets/ZeroLine/ZeroLine_' + str(number) + '.txt'))

    for interval in sample.data:
        print(interval)

    sample.draw_sample_plot('Interval sample')
    coeffs = UtilHelper.calculate_regression(sample)
    print(f'beta_0 = {coeffs[0]}, beta_1 = {coeffs[1]}')
    UtilHelper.plot_regression(coeffs, sample, 'Point regression of interval sample')
    inform_set = UtilHelper.build_inform_set(sample, coeffs, plot=True)
    UtilHelper.plot_corridor(sample, inform_set)


if __name__ == '__main__':
    main()
