# Rigveda data

## pada-and-analysis.dat

This file contains the padas of the Rigveda (without accents) and the corresponding analyses from the DCS.
Each line represents one pada.
The individual pieces of information are separated with tabulators (\t):
* book, hymn, stanza, pada: The citation
* text: Full text of the pada without accents
* lemmata: The lemmata as strings; separated with spaces (' ')
* lexids: = DCS.lexicon.id (or: DCS.word_references.lexicon_id), i.e. the id of the lemma; separated with spaces (' ')
* refids: = DCS.word_references.id, i.e. the id of the occurrence; separated with spaces (' ')

The mapping was performed automatically, so there may be errors in the data file.
These errors are recorded in ''pada-and-analysis.err''.