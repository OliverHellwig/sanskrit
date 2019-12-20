# DCS in conllu format

## General notes

This is a dump of most of the texts in the DCS in a human readable, tab separated format. Most commentarial texts are not included in this dump.
The analysis of each string has been checked by one annotator.

The subdirectory ./files contains the texts in an extended CoNLL-U format (see https://universaldependencies.org/format.html).
Current size of the dump:

* Number of lines: 620155
* Number of words: 4699761

Texts shorter than 50,000 words are stored in individual files in the directory ./files. Their names end with -all (= containing all chapters).
Longer texts are split into one file per chapter and stored in subdirectories of ./files.
The file name "Aṣṭāṅgahṛdayasaṃhitā-0007-AHS, Sū., 8-1162" means: Text=Aṣṭāṅgahṛdayasaṃhitā", chapter seven ("0007"), citation form of the chapter name="AHS, Sū., 8", chapter id="1162"

Each line starting with a number contains twelve entries. The following list explains entries whose meaning is not the same as in the UD standard as well as the additional fields (11++).

2. FORM: Word form or punctuation symbol. - A single string (= sequence of letters limited by spaces) can contain one or multiple words in Sanskrit. If it contains multiple words, the annotation follows the proposals for multiword annotation (https://universaldependencies.org/format.html#words-tokens-and-empty-nodes).
3. LEMMA: Lemma or stem of word form. - The lexical id of the lemma is found in field 11.
4. UPOS: Universal part-of-speech tag. - Not set.
5. XPOS: Language-specific part-of-speech tag. - This tag set is described in ./files/lookup/pos.csv and Hellwig, Hettrich, Modi and Pinkal (2018): Multi-layer Annotation of the Rigveda.
11. Lemma ID, numeric. Matches the first column of ./files/lookup/dictionary.csv
12. The unsandhied word form ("padapāṭha version", univerbified preverbs are not resolved, but can be retrieved from the column 'preverbs' in ./files/lookup/dictionary.csv); only available for a part of the DCS, mainly the Vedic texts.

## Directory 'lookup'

This directory contains lexicographic information (dictionary.csv) and a short explanation of the POS tags (pos.csv).

## Citation

If you use these data in your work, please use this citation:

Oliver Hellwig: Digital Corpus of Sanskrit (DCS). 2010-2019.

In bibtex:
```
@Manual{your_latech_key,
title = {{The Digital Corpus of Sanskrit (DCS)}},
author = {Hellwig, Oliver},
year = {2010--2019}
}
```

 
## License

The data in this directory are licensed under the Creative Common BY 4.0 (CC BY 4.0) license.