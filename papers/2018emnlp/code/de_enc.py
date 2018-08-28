import defines
import codecs,os

class DeEncoder(object):
    def __init__(self):
        '''
        input character => numeric index
        '''
        self.sym2idx = {}
        self.idx2sym = []
        self.get_index(defines.SYM_PAD)
        self.get_index(defines.SYM_UNK)
        self.get_index(defines.SYM_SPACE)
    def get_index(self, sym, freeze = False, allow_unk = False):
        '''
        @param freeze: If true, don't add new symbols to this lookup (exc. if allow_unk == True) 
        '''
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        if freeze == True:
            if allow_unk==True:
                if not defines.SYM_UNK in self.sym2idx:
                    self.sym2idx[defines.SYM_UNK] = len(self.sym2idx)
                    self.idx2sym.append(defines.SYM_UNK)
                return self.sym2idx[defines.SYM_UNK]
            raise ValueError('trying to add a new symbol to a frozen encoder')
        idx = len(self.sym2idx)
        self.sym2idx[sym] = idx
        self.idx2sym.append(sym)
        return idx
    def get_sym(self, idx):
        if idx < len(self.idx2sym):
            return self.idx2sym[idx]
        print("unknown index: {0}".format(idx))
        return self.idx2sym[0] # UNK
    def get_size(self):
        return len(self.idx2sym)
    def store(self, path):
        with codecs.open(path, 'w', 'UTF-8') as f:
            for i in range(len(self.idx2sym)):
                f.write('{0}\n'.format(self.idx2sym[i]))
    def load(self, path):
        # reset these lists, because the ctr adds SPACE by default
        self.sym2idx = {}
        self.idx2sym = []
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with codecs.open(path, 'r', 'UTF-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    ix = len(self.idx2sym)
                    self.idx2sym.append(line)
                    self.sym2idx[line] = ix
    def build(self, idx2sym_):
        '''
        @param idx2sym_: array with symbols 
        '''
        self.sym2idx = {}
        self.idx2sym = []
        for ix, sym in enumerate(idx2sym_):
            self.idx2sym.append(sym)
            self.sym2idx[sym] = ix