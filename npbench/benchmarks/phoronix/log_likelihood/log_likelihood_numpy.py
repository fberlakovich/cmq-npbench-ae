#from: http://arogozhnikov.github.io/2015/09/08/SpeedBenchmarks.html
import numpy

#pythran export log_likelihood(float64[], float64, float64)
def log_likelihood(data, mean, sigma):
    s = (data - mean) ** 2 / (2 * (sigma ** 2))
    pdfs = numpy.exp(- s)
    pdfs /= numpy.sqrt(2 * numpy.pi) * sigma
    return numpy.log(pdfs).sum()
