# DCS Data Dump

The current (Nov. 2018) dump of the DCS main database can be found on [Google Drive](https://drive.google.com/open?id=1zKHtrnRTqW6TroOoepFgTGBsPT9D6i6k).

Most relational constraints are indicated in the column comments of the tables.
I tried hard (but not always successfully) to follow a consistent naming convention for foreign keys; so, the column 'meaning.lexicon_id' links to 'lexicon.id'.
An important deviation from this scheme is the link between the column 'word_references.sentence_id' (where a certain lemma is found in a line of text) and 'text_lines.id'.

If a column comment contains the term 'permanent', the value of this column should not change between different dumps of the database; see, for example, the column 'lexicon.id', which gives permanent identifiers of each word.

## License

The data of the DCS and any data in child directories are licensed under the Creative Common BY 4.0 (CC BY 4.0) license.

## Citation

If you use the DCS data in your work, please use the following citation:

Oliver Hellwig: Digital Corpus of Sanskrit (DCS). 2010-2021.

In bibtex:
```
@Manual{your_latech_key,
title = {{The Digital Corpus of Sanskrit (DCS)}},
author = {Hellwig, Oliver},
year = {2010--2021}
}
```

