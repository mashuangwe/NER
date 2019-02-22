class TmpTest:
    def __init__(self, filename):
        self.filename = filename
        print('__init__')

    def __enter__(self):
        self.f = open(self.filename, 'r')
        print('__enter__')
        return self.f

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('__exit__ {}'.format(exc_type))
        self.f.close()
        return True

test = TmpTest('file')
with test as f:
    print('test result: {}'.format(t))
    raise ImportError
print('no error')