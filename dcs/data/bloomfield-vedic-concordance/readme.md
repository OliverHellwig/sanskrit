Bloomfield's Vedic Concordance ([see here](http://www.people.fas.harvard.edu/~witzel/VedicConcordance/ReadmeEng.html) for the original data and the copyright notice) is extremely useful, but can be difficult to parse in its original format.
It's also complicated to find text references consumed in lists and enumerations.
The entry ŚB.3.2.2.20-5.20, for example, can be expanded into ŚB.3.2.2.20; ŚB.3.2.3.20; ŚB.3.2.4.20; ŚB.3.2.5.20.
After the transformation, you can easily find the reference *ŚB.3.2.4.20*, which was not present in this form in the original format.

This directory contains a Python script that performs (most of) these conversions.
The transformed output is found in data/bloomfield-vc-full.txt, along with various other formats.

Each line in data/bloomfield-vc-full.txt is structured by the sequence ' $ '.
To the left of this separator, you find the original line, and to its right the output of the script with resolved references.

**Example:**
*Original line (no. 375)*: 
>agnaye kāmāya svāhā # TB.3.12.2.2-8. Cf. kāmāya svāhā, and agnīṣomābhyāṃ kāmāya.

*Transformed line*:
>agnaye kāmāya svāhā # TB.3.12.2.2-8. Cf. kāmāya svāhā, and agnīṣomābhyāṃ kāmāya.** $ **>agnaye kāmāya svāhā # TB.3.12.2.2; TB.3.12.2.3; TB.3.12.2.4; TB.3.12.2.5; TB.3.12.2.6; TB.3.12.2.7; TB.3.12.2.8.

After the separator ' $ ', the reference *TB.3.12.2.2-8.* has been resolved into TB.3.12.2.2; TB.3.12.2.3; TB.3.12.2.4; TB.3.12.2.5; TB.3.12.2.6; TB.3.12.2.7; TB.3.12.2.8.
All additional information has been removed from the second part, but can still be found in the first one.

The file *bloomfield-vc.json* contains the data in JSON format; the files *bloomfield-vc-R-mantras.dat* and *bloomfield-vc-R-citations.dat* (column `mantra_id' = bloomfield-vc-R-mantras::id) contain them as data frames that can be imported into R or similar programs.

Currently (Jan. 2020), the final script produces less than 1,000 error messages (inspect data/errs.dat after running the script).
Most of them result from struggling with an unexpectedly formatted line (strategy (1): adapt the script; (2) manually change the source file).