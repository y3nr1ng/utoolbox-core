class Tiff(object):
    def __init__(self, name):
        from os.path import basename
        print('file \'{}\''.format(basename(name)))
