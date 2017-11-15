from utoolbox.io import TimeSeries
import utoolbox.io.primitives as dtype

ts = TimeSeries(dtype.SimpleVolume, folder='data')
print('{} time points'.format(len(ts)))
