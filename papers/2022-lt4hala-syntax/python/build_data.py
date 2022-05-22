#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
next step: R
'''

import codecs,re

def splitJointmorph(jointMorph):
    cas = '_'
    num = '_'
    gen = '_'
    verbform = '_'
    tense = '_'
    if jointMorph!='_':
        t5 = jointMorph.strip()
        ms = t5.split('|')
        for m in ms:
            if m.startswith('Case='):
                cas = m[5:]
            elif m.startswith('Number='):
                num = m[7:]
            elif m.startswith('Gender='):
                gen = m[7:]
            elif m.startswith('VerbForm='):
                verbform = m[9:]
            elif m.startswith('Tense='):
                tense = m[6:]
        if cas=='_':
            if 'Mood' in t5:
                cas = 'FV'
            elif 'VerbForm' in t5:
                cas = 'IV'
            else:
                cas = 'ind'
    return cas,num,gen,verbform,tense

conlluPath = '../data/sanskrit.conllu'
senNo = 1
wrdNo = 1
sep = '\t'
with codecs.open(conlluPath, 'r', 'UTF-8') as fIn, codecs.open('../data/conllu.dat','w','UTF-8') as fOut:
    hdr = ['text','chapter','layer','sen.id','wrd.position','head.position','word','lemma','lemma.id','pos','label','sublabel','cas','num','gen','verbform','tense','annotator']
    fOut.write(sep.join(hdr) + '\n')
    reSenId = re.compile('^# sent_id = (.+)$')
    reSenLayer = re.compile('^# layer=(.+)$')
    written = 0
    for lnum,line in enumerate(fIn):
        line = line.strip()
        if not line:
            continue
        m = reSenId.match(line)
        if m:
            '''
            main headline of one sentence
            '''
            if written>0:
                sen = []
                senNo+=1
                wrdNo =1
            cit = m.group(1)
            ix = cit.find('_')
            txtname = cit[:ix]
            chapter = cit[(ix+1):]
            layer = ''
            continue
        if line.startswith('#'): # full text
            m = reSenLayer.match(line)
            if m:
                s = m.group(1)
                layer = s[:s.find('-')]
            continue
        '''
        this is a regular line
        '''
        toks = line.split('\t')
        if len(toks) < 12:
            print('line with {0} tokens'.format(len(toks)))
            continue
        wrd = toks[2] # lexicon entry!
        surfaceWrd = toks[1] if toks[1]!='_' else '[{0}]'.format(toks[2])
        lemmaId = toks[14]
        posTag = toks[3]
        headPosition = toks[6]
        
        labtoks = toks[7].split(':')
        label = labtoks[0]
        sublabel = labtoks[1] if len(labtoks)==2 else ''
        annotator = toks[11]
        ''' morpho-syntax '''
        jointMorph = toks[5]
        cas,num,gen,verbform,tense = splitJointmorph(jointMorph)
        
        ''' write to the out file '''
        itms = [txtname,chapter,layer,senNo,wrdNo,headPosition,surfaceWrd,wrd,lemmaId,posTag,label,sublabel,cas,num,gen,verbform,tense,annotator]
        fOut.write(sep.join([str(x) for x in itms]) + '\n')
        written+=1
        wrdNo+=1
