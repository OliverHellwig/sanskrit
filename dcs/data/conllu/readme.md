# DCS in conllu format

## General notes

This is a dump of most texts in the DCS in a human-readable, tab-separated format. Most commentary texts are not included in this dump.
The analysis of each string has been verified by one annotator.

The subdirectory ./files contains the texts in CoNLL-U format (see https://universaldependencies.org/format.html). Each text is in a separate folder, and each chapter (i.e. what is defined as a chapter in the DCS) is in a separate file.
The file name "Aṣṭāṅgahṛdayasaṃhitā-0007-AHS, Sū., 8-1162" means: Text="Aṣṭāṅgahṛdayasaṃhitā", chapter seven ("0007"), citation form of the chapter name="AHS, Sū., 8", chapter ID="1162"
Note that the format of the CoNLL-U files has changed in the Aug 9, 2022 release and now complies with the UD standard.

Current size of the dump:

* Number of lines: 744,757
* Number of words: 5,464,818

Each line that begins with a number contains ten entries. The following list explains entries whose meanings differ from the UD standard.


																																																  

* 2. FORM: Word form or punctuation symbol. - A single string (= sequence of letters bounded by spaces) can contain one or multiple words in Sanskrit. If it contains multiple words, the annotation follows the guidelines for multiword annotation (https://universaldependencies.org/format.html#words-tokens-and-empty-nodes).
* 3. LEMMA: Lemma or stem of word form. - The lexical ID of the lemma is found in field 11.
* 4. UPOS: Universal part-of-speech tag. - The value of this field is automatically mapped from the XPOS field.
* 5. XPOS: Language-specific part-of-speech tag. - This tag set is described in ./files/lookup/pos.csv and Hellwig, Hettrich, Modi and Pinkal (2018): Multi-layer Annotation of the Rigveda.
* 7. HEAD: If the chapter is part of the Vedic Treebank, this field contains the line number (= column 1) of the syntactic head.
* 8. DEPREL: If the chapter is part of the Vedic Treebank, this field contains the label of the syntactic arc.
* 10. MISC: This field can contain any combination of the following key-value pairs:
  * LemmaId: Matches the first column of ./files/lookup/dictionary.csv
  * OccId: The ID of this occurrence of the word
  * Unsandhied: The unsandhied word form, "padapāṭha version". Univerbified preverbs are not resolved but can be retrieved from the column 'preverbs' in ./files/lookup/dictionary.csv; only available for a part of the DCS.
  * WordSem: IDs of lexical semantic concepts. The IDs correspond to the first column in ./files/lookup/word-senses.csv. A more user-friendly version of this data is available through the [Sanskrit WordNet](http://sanskritwordnet.unipv.it/).
  * Punctuation ['comma', 'fullStop']: Not part of the original Sanskrit text, but inserted in a separate layer.
  * IsMantra: True if this word is part of a mantra as recorded in Bloomfield's Vedic Concordance.


## Directory 'lookup'

This directory contains lexicographic information (dictionary.csv), details about word senses (subfield WordSem in field 10 of a conllu line; file word-senses.csv) and a brief explanation of the POS tags (pos.csv).

Additionally, the file 'sembank-relations.csv' contains the operators needed to reconstruct the WordNet underlying the lexical semantic annotation in the DCS. Symbols used are: "~" for subclasses, "i" for instances, "p" for parts, and "m" for members of a class. An overview of the attestations of lexical senses is provided in 'sembank-attestations.csv'.

## Citation

If you use these data in your work, please cite as follows:


```
@Manual{your_latech_key,
title = {{The Digital Corpus of Sanskrit (DCS)}},
author = {Hellwig, Oliver},
year = {2010--2024}
}
```

 
## License

The data in this directory are licensed under the Creative Commons BY 4.0 (CC BY 4.0) license.