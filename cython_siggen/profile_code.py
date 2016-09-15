import cProfile
import pstats

#from test_detector import test_detector
#cProfile.run('test_detector()', 'output.txt')

from fit_detector_nofields import main
cProfile.run('main()', 'output.txt')

p = pstats.Stats('output.txt')
p.strip_dirs().sort_stats(-1).print_stats()

p.sort_stats('cumulative')
p.print_stats(20)