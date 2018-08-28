

class SplitCounter(object):
    def __init__(self):
        pass
    def join_nums(self, vals):
        '''
        This should speed up the n-gram building ~ by the factor 3 when compared with
        
            ' '.join([str(n) for n in vals])
        '''
        L = len(vals)
        if L==2:
            return '{0} {1}'.format(vals[0], vals[1])
        elif L==3:
            return '{0} {1} {2}'.format(vals[0], vals[1], vals[2])
        elif L==4:
            return '{0} {1} {2} {3}'.format(vals[0], vals[1], vals[2], vals[3])
        elif L==5:
            return '{0} {1} {2} {3} {4}'.format(vals[0], vals[1], vals[2], vals[3], vals[4])
        elif L==6:
            return '{0} {1} {2} {3} {4} {5}'.format(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])
        elif L==7:
            return '{0} {1} {2} {3} {4} {5} {6}'.format(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6])
        return ' '.join([str(x) for x in vals])
