from lattice.io import TimeSeries

ts = TimeSeries('SimpleVolume', folder='data')
print('{} time points'.format(len(ts)))
