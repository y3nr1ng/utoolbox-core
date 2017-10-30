from lattice.io import TimeSeries
import lattice.io.primitives as dtype

ts = TimeSeries(dtype.SimpleVolume, folder='data')
print('{} time points'.format(len(ts)))
