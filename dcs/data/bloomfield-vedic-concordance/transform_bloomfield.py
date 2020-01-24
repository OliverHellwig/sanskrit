'''
aim: Prepare Bloomfield for easier processing.
Apply a number of regexes that clean up the format.
Ideally, only cited passages and full references should be contained in the output.
@author: Oliver Hellwig <hellwig7@gmx.de>
@todo: Some cases are currently not handled (see data/errs.dat); and some regexes are not beautiful.
'''
import codecs, re
restrs = [
    #['^.+ # [^ ]+$' , ''],
    ['^(.+) \\([^#]+?\\) (.+)$', '\\1 \\2'], # >am̐śuṃ gabhasti (KS. babhasti) haritebhir āsabhiḥ # => >am̐śuṃ gabhasti haritebhir āsabhiḥ #
    #  MS.2.6.11: 70.9;  =>  MS.2.6.11
    ['^(.+) ([^ \:]+)\: {0,1}[0-9\.\,]+(\. |; )(.+)$', '\\1 \\2\\3\\4'],
    # MS.4.14.15b: 240.5.[EOL]
    ['^(.+) ([^ \\:]+)\\: {0,1}[0-9\,\.]+\.$', '\\1 \\2.'],
    # RV.1.113.16d; 8.48.11d; => RV.1.113.16d; RV.8.48.11d;
    ['^(.+) ([^0-9 \.]+)\.([0-9]+\.[0-9]+\.[0-9]+[a-z]*); ([0-9]+\.[0-9]+\.[0-9]+[a-z]*)((\. |\,|; ).+)$', '\\1 \\2.\\3; \\2.\\4\\5'],
    ['^(.+) ([^0-9 \.]+)\.([0-9]+\.[0-9]+\.[0-9]+[a-z]*); ([0-9]+\.[0-9]+\.[0-9]+[a-z]*)\.$', '\\1 \\2.\\3; \\2.\\4.'], # EOL
    ### ---- 4 elements
    # 4 -> 4
    ## TS.4.3.6.1a; 5.3.2.1; => TS.4.3.6.1a; TS.5.3.2.1;
    ['^(.+) ([^0-9 \.]+)((\.[0-9]+){4}[a-z]*); ([0-9]+)((\.[0-9]+){3}[a-z]*)((\. |,|;).+)$', '\\1 \\2\\3; \\2.\\5\\6\\8'],
    ['^(.+) ([^0-9 \.]+)\.([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+[a-z]*); ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+[a-z]*)\.$', '\\1 \\2.\\3; \\2.\\4.'], # EOL
    # TS.1.6.2.4; 11.6; OR # ŚB.4.2.2.16; 3.10,15-17 
    ['^(.+) ([^0-9 \.]+)\.(([0-9]+\.){2})([0-9]+\.[0-9]+[a-z]*); ([0-9]+\.[0-9]+[a-z]*)([\.;]) (.+)$', '\\1 \\2.\\3\\5; \\2.\\3\\6\\7 \\8'],
    ['^(.+) ([^0-9 \.]+)\.([0-9]+\.[0-9]+\.)([0-9]+\.[0-9]+[a-z]*); ([0-9]+\.[0-9]+[a-z]*)\.$', '\\1 \\2.\\3\\4; \\2.\\3\\5.'], # EOL
    ['^(.+) ([^0-9 \.]+(\.[0-9]+){2})((\.[0-9]+){2}[a-z]*); ([0-9]+\.[0-9]+[a-z]*)(,[0-9].+)$', '\\1 \\2\\4; \\2.\\6\\7'], # with comma
    
    # 4 -> 3
    ## ŚB.4.1.5.17; 2.1.9,12;
    ['^(.+) ([^0-9 \.]+\.[0-9]+)((\.[0-9]+){3}[a-z]*); (([0-9]+\.){2})([0-9]+[a-z]*)([\.;] .+)$', '\\1 \\2\\3; \\2.\\5\\7\\8'],
    ['^(.+) ([^0-9 \.]+\.[0-9]+)((\.[0-9]+){3}[a-z]*); (([0-9]+\.){2})([0-9]+[a-z]*)(,[0-9].+)$', '\\1 \\2\\3; \\2.\\5\\7\\8'], # with comma
    ## TB.3.8.17.2; 12.4.2-6;
    ['^(.+) ([^0-9 \.]+\.[0-9]+)((\.[0-9]+){3}[a-z]*); ([0-9]+(\.[0-9]+){2}[a-z]*\-[0-9]+[a-z]*)((\.\b|,|;).*)$', '\\1 \\2\\3; \\2.\\5\\7\\8'],
    # 4 -> 2
    # ŚB.4.5.3.10; 4.9-11; => ŚB.4.5.3.10; ŚB.4.5.4.9-11; 
    ['^(.+) ([^0-9 \.]+)((\.[0-9]+){2})(\.[0-9]+){2}; ([0-9]+\.[0-9]+\-[0-9]+[a-z]*)((\. |,|;).+)$', '\\1 \\2\\3\\5; \\2\\3.\\6\\7'],
    ## TS.1.4.13.1; 15.1-21.1; 
    ['^(.+ )([^0-9 \.]+(\.[0-9]+){2})(\.[0-9]+){2}([a-z]*); ([0-9]+\.[0-9]+\-[0-9]+\.[0-9]+[a-z]*)((\.\b|,|;).+)$', '\\1\\2\\4\\5; \\2.\\6\\7'],
    # AB.2.40.7; 41.9; => AB.2.40.7; AB.2.41.9;
    ['^(.+) ([^0-9 \.]+)\.([0-9]+)\.([0-9]+\.[0-9]+[a-z]*); ([0-9]+\.[0-9]+[a-z]*)([\.;]) (.+)$', '\\1 \\2.\\3.\\4; \\2.\\3.\\5\\6 \\7'],
    ['^(.+) ([^0-9 \.]+)\.([0-9]+)\.([0-9]+\.[0-9]+[a-z]*); ([0-9]+\.[0-9]+[a-z]*(\-[0-9]+[a-z]*){0,1}),(.+)$', '\\1 \\2.\\3.\\4; \\2.\\3.\\5,\\7'],# with following range
    # RV.6.25.4c; 66.8c.[EOL]
    ['^(.+) ([^0-9 \.]+)\.([0-9]+)\.([0-9]+\.[0-9]+[a-z]*); ([0-9]+\.[0-9]+[a-z]*)\.$', '\\1 \\2.\\3.\\4; \\2.\\3.\\5'],
    # VS.8.4d; 33.68d; + EOL
    ['^(.+) ([^0-9 \.]+)\.([0-9]+\.[0-9]+[a-z]*); ([0-9]+\.[0-9]+[a-z]*(\-[0-9]+[a-z]*){0,1})([\.;]) (.+)$', '\\1 \\2.\\3; \\2.\\4\\6 \\7'],
    ['^(.+) ([^0-9 \.]+)\.([0-9]+\.[0-9]+[a-z]*); ([0-9]+\.[0-9]+[a-z]*)(\,)(.+)$', '\\1 \\2.\\3; \\2.\\4\\5\\6'],
    ['^(.+) ([^0-9 \.]+)\.([0-9]+\.[0-9]+[a-z]*); ([0-9]+\.[0-9]+[a-z]*)\.$', '\\1 \\2.\\3; \\2.\\4.'], # EOL
    # KS.19.11,12; => KS.19.11; KS.19.12; AND AVP.4.38.1b,2a. => AVP.4.38.1b; AVP.4.38.2a.
    ['^(.+) ([^0-9 \.]+)\.(([0-9]+\.){1,4})([0-9]+[a-z]*),([0-9]+[a-z]*(\-[0-9]+[a-z]*){0,1})([\.;\,])(.+)$', '\\1 \\2.\\3\\5; \\2.\\3\\6\\8\\9'],
    ['^(.+) ([^0-9 \.]+)\.(([0-9]+\.){1,4})([0-9]+[a-z]*),([0-9]+[a-z]*)\.$', '\\1 \\2.\\3\\5; \\2.\\3\\6.'], # EOL
    # special stuff
    # HirŚ.5.4.82:486.
    #['^(.+ )(HirŚ(\.[0-9]+){1,3}[a-z]*)\:[0-9]+([\.;])(.*)$', '\\1\\2\\4\\5']
    ]

res = [re.compile(s[0]) for s in restrs ]
# AVŚ.3.26.1-6
re_range1 = re.compile('^(.+ )([^0-9 \.]+)\.(([0-9]+\.){0,3})([0-9]+)[a-z]{0,1}\-([0-9]+)[a-z]{0,1}((\. |\.$|;|\,).*)$')
# TS.1.4.2.1-13.1
# todo PCRE version, use python::regex?
#re_range2 = re.compile('^(.+ )([^0-9 \.]+)\.(([0-9]+\.){0,3})\.([0-9]+)\.([0-9]+)[a-z]{0,1}\-([0-9]+)\.(?6)[a-z]{0,1}((\.\b|;|\,).*)$')
re_range2 = re.compile('^(.+ )([^0-9 \.]+(\.[0-9]){0,3})\.([0-9]+)\.([0-9]+)[a-z]{0,1}\-([0-9]+)\.([0-9]+)[a-z]{0,1}((\.\b|;|\,).*)$') 
# ApMB <-> ApG
re_ApG = re.compile('^(.+ )(ApMB(\.[0-9]+){2,3}[a-z]{0,1}) \(ApG(\.[0-9]+){2,3}[a-z]{0,1}\)(.+)$' )
re_brackets = re.compile('^(.+) # (.+) \([^\)]+?\)(.*)$')
# Transform xxx Ps: aaa bbb; ccc ddd into xxx P: aaa bbb; P: ccc ddd
re_pratikas = re.compile('^(.+ # .+) Ps\:( .+)$')
nagari = 'aāiīuūṛeokgṅcjñtdnṭḍṇpbmyrlvśsṣhṁṃḥ'
re_pratika_details = re.compile("^([{0} \-']+) ([^{0}].+)$".format(nagari))
re_occ = re.compile("^([^0-9 \.]+)(\.[0-9]+)+[A-Da-z]{0,2}$") # should match AVŚ.3.26.1ab, but not AVŚ.3.26.1-6 etc.
#remove_single_quots = False

tstln = '>aṃhoś cid yā varivovittarāsat # RV.1.107.1d; VS.8.4d; 33.68d; TS.1.4.22.1d; 2.1.11.4d; MS.1.3.26d: 39.8; KS.4.10d; ŚB.4.3.5.15d.'
mantras = [] # for json
with codecs.open('./data/bloomfield-vc.txt', 'r', 'UTF-8') as fin,\
    codecs.open('./data/bloomfield-vc-transformed.txt', 'w', 'UTF-8') as fout,\
    codecs.open('./data/bloomfield-vc-full.txt', 'w', 'UTF-8') as ffull,\
    codecs.open('./data/errs.dat', 'w', 'UTF-8') as ferr:
    for lineno,line in enumerate(fin):
        if lineno<=3: # preamble
            fout.write(line)
            ffull.write(line)
            continue
        line = line.strip()
        line_orig = line
        if re.match('^.+ # see .+$', line):
            fout.write('\n')
            ffull.write('{0} $ \n'.format(line_orig))
            continue
        #line = tstln
        # some texts are not correctly formatted
        line = line.replace('BhārŚ ', 'BhārŚ.').replace('ApŚ ', 'ApŚ.'). replace('VaikhŚ ', 'VaikhŚ.')
        #line = line.replace('\:[0-9]+?')
        # ApMB.2.18.16 (ApG.7.20.4) => ApMB.2.18.16 (see VC, p. xix; the reference in the ApG describes the application of the mantra)
        ma = re_ApG.match(line)
        while ma:
            line = ma.group(1) + ma.group(2) + ma.group(5)
            ma = re_ApG.match(line)
        # remove everything in round brackets
        ma = re_brackets.match(line)
        while ma:
            line = '{0} # {1}{2}'.format(ma.group(1),ma.group(2), ma.group(3))
            ma = re_brackets.match(line)
        for s in [' See ', ' Cf. ', ' Designated as ', ' Vikāra of ', ' Variations of ', ' Variation of ']:
            ix = line.find(s)
            if ix>-1:
                line = line[:ix].strip()
        line = line.replace(' Fragment: ', ' P: ') # fragment = pratika
        # one citation only?
#         if remove_single_quots==True and re.match('^.+ # [^ ]+$', line): # mantra text and a single occurrence ... may be useful, however
#             fout.write('\n')
#             ffull.write('{0} $ {1}\n'.format(line,line_orig))
#             continue
        '''pratikas, Ps.
        strategy: replace " Ps.: text1; ref1; text2; ref2" with " P: text1; ref1; P: text2; ref2"
        '''
        ma = re_pratikas.match(line)
        if not ma is None:
            rsplit = [s for s in re.split(" ([aāiīuūṛeokgṅcjñtdnṭḍṇpbmyrlvśsṣhṁṃḥ \-']+) ", ma.group(2)) if s!=""]
            if len(rsplit)%2==0:
                line = ma.group(1) + ' ' + ' '.join(['P: {0} {1}'.format(u,v) for u,v in zip(rsplit[0::2],rsplit[1::2]) ])
        ln = ''
        while line!=ln:
            ln=line
            # special treatment of ranges
            # var. 1: ^(.+ )([^0-9 \.]+)\.(([0-9]+\.){0,3})([0-9]+)[a-z]{0,1}\-([0-9]+)[a-z]{0,1}((\.\b|;|\,).*)$
            ma = re_range1.match(line)
            while ma:
                start = int(ma.group(5))
                end   = int(ma.group(6))+1
                txt = ''
                for x in range(start,end):
                    txt+='; ' if x>start else ''
                    txt+='{0}.{1}{2}'.format(ma.group(2),ma.group(3),x)
                line = ma.group(1) + txt + ma.group(7)
                ma = re_range1.match(line)
            # var. 2: ^(.+ )([^0-9 \.]+(\.[0-9]){0,3})\.([0-9]+)\.([0-9]+)[a-z]{0,1}\-([0-9]+)\.([0-9]+)[a-z]{0,1}((\.\b|;|\,).*)$
            ma = re_range2.match(line)
            lstart = 0
            while ma:
                u=ma.group(5); v=ma.group(7)
                if ma.group(5)==ma.group(7):
                    start = int(ma.group(4))
                    end   = int(ma.group(6))+1
                    txt = ''
                    for r in range(start,end):
                        txt+='; ' if r>start else ''
                        txt+='{0}.{1}.{2}'.format(ma.group(2),r,ma.group(5))
                    line = ma.group(1) + txt + ma.group(8)
                else:
                    print('invalid range in line no {0}: '.format(lineno+1) )
                    break
                ma = re_range2.match(line)
            # the other regexes
            for i,reg in enumerate(res):
                while reg.match(line):
                    line = re.sub(restrs[i][0], repl=restrs[i][1], string=line).strip()
                    #print('{0} {1}'.format(i,line))
        if line!='':
            fout.write(line+'\n')
            ffull.write('{0} $ {1}\n'.format(line_orig, line))
        # prepare some more structured output
        toks = line.split(' # ')
        if len(toks)==2:
            # before splitting by ; , we need to handle the pratIka stuff ( P: )
            mantra = [toks[0].strip().lstrip('>'), [] ] # [om om om svAhA -> [[AB.1.3.5, ''], [AB.2.3.5, 'om om']]]
            Owp = re.split(' P: ', toks[1].strip())
            occs_with_pratika = []
            for owp in Owp:
                ma = re_pratika_details.match(owp.strip())
                if ma:
                    occs_with_pratika.append([ma.group(2).strip(), ma.group(1).strip()])
                else:
                    occs_with_pratika.append([owp.strip(), ''])
            for (o,pr) in occs_with_pratika:
                oo = o.split('; ')
                for j,ooo in enumerate(oo):
                    ooo = ooo.rstrip('.').rstrip(';')
                    # this can be critical in cases such as:
                    # "ŚŚ.1.13.3. The passage seems metrical: pādas after apiprer, amatsata, devaṃgamāṃ"
                    if not re_occ.match(ooo):
                        ix = ooo.find('. ')
                        if ix>-1:
                            ooo = ooo[:ix]
                    if not re_occ.match(ooo):
                        ferr.write('ln {0} non-matching citation: {1}\n'.format(lineno+1,ooo))
                        continue
                    mantra[1].append([ooo,pr])
            if len(mantra[1])>0:
                mantras.append(mantra)
        #break # debug
with codecs.open('./data/bloomfield-vc.json', 'w', 'UTF-8') as f,\
    codecs.open('./data/bloomfield-vc-R-mantras.dat', 'w', 'UTF-8') as fRma,\
    codecs.open('./data/bloomfield-vc-R-citations.dat', 'w', 'UTF-8') as fRcit:
    fRma.write('id\tmantra\n' + '\n'.join('{0}\t{1}'.format(i+1,ma[0]) for i,ma in enumerate(mantras)) )
    sep = '\t'
    fRcit.write(sep.join(['txt1', 'lvl11', 'lvl12', 'lvl13', 'lvl14',  'txt2', 'lvl21', 'lvl22', 'lvl23', 'lvl24' , 'mantra_id', 'pratika' ]) + '\n')
    f.write('[')
    for i,ma in enumerate(mantras):
        affix = ',' if i<(len(mantras)-1) else ''
        f.write('\n\t{{"mantra":"{0}",\n\t\t"cits":[{1}]}}{2}'.format(ma[0], ','.join('{{"cit":"{0}","p":"{1}"}}'.format(r,p) for (r,p) in ma[1]), affix) )
        # R
        if len(ma[1])>1:
            tmp = []
            for (t,p) in ma[1]: # for all citations ...
                toks = t.split('.')
                if len(toks)<=5:
                    txt = toks[0]
                    lvls = toks[1:]
                    while len(lvls) < 4:
                        lvls.insert(0,'')
                    tmp.append([sep.join([txt] + lvls), p])
                else:
                    print('len toks: {0}'.format(len(toks)))
            for j in range(2,len(tmp)):
                fRcit.write(sep.join(str(x) for x in [tmp[0][0], tmp[j][0], (i+1), tmp[j][1]]) + '\n')
    f.write('\n]') 