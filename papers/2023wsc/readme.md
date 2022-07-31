# A new XML-based version of Bloomfield's Vedic Concordance

This directory contains data and code for our paper at the World Sanskrit Conference 2023:

```
Oliver Hellwig, Sven Sellmer and Kyoko Amano: The Vedic corpus as a graph. An updated version of Bloomfield’s Vedic Concordance
```

The main contribution is found in data/vc.xml. This file is a new and extended version of Bloomfield's well known Vedic Concordance, based on the electronic edition by Franceschini (2007). 
Bloomfield's Vedic Concordance ([see here](http://www.people.fas.harvard.edu/~witzel/VedicConcordance/ReadmeEng.html) for the original data and the copyright notice) is extremely useful, but can be difficult to parse in its original format.
It is also complicated to find text references consumed in lists and enumerations.
The entry ŚB.3.2.2.20-5.20, for example, can be expanded into ŚB.3.2.2.20; ŚB.3.2.3.20; ŚB.3.2.4.20; ŚB.3.2.5.20.
Our aim was to explicitly encode such kind of information found in previous printed and electronic versions of this work. This includes, among other, resolving all citations and variants. Detailed information is given in the paper (directory paper/).

Entries from the following texts have been added to the Vedic Concordance:

<table>
	<thead>
	<tr>
		<th>Text</th> <th>Number of entries</th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td>Bhāradvājaśrautasūtra (BhārŚ)</td> <td>2094</td>
	</tr>
	<tr>
		<td>Baudhāyanaśrautasūtra (BaudhŚ)</td> <td>8930</td>
	</tr>
	<tr>
		<td>Baudhāyanagṛhyasūtra (BaudhGS)</td> <td>1325</td>
	</tr>
	<tr>
		<td>Kāṭhakagṛhyasūtra (KāṭhGS)</td> <td>605</td>
	</tr>
	<tr>
		<td>Vaikhānasaśrautasūtra (VaikhŚ)</td> <td>1169</td>
	</tr>
	<tr>
		<td>Śāṅkhāyana-Āraṇyaka (ŚāṅkhĀ)</td> <td>230</td>
	</tr>
	</tbody>
</table>

The large majority of them were classified as citation (type="default"), variant (type="variant"), pratīka (type="pratika"), or fragment (type="fragment"). Only few (332) completely new main entries were created. As a rule, these contain mantras that are closely related to mantras already contained in the VC, but not in the way of variants, pratīkas, or fragments. Instead, these mantras exchange one element for another one in a systematic manner (e.g., in a series of mantras addressing different gods) and could therefore also be labeled as ūhas. 