# Sanskrit Sandhi and Compound Splitter

This directory contains code and data for the following paper:

Oliver Hellwig, Sebastian Nehrdich: Sanskrit Word Segmentation Using Character-level Recurrent and
Convolutional Neural Networks. In: Proceedings of the EMNLP 2018.

Code and data are licensed under the GNU AGPLv3 license.

## Applying a pre-trained model to local data

The folder data/models contains a pre-trained model that can be loaded locally and applied to un-sandhied text files.

In the best of all worlds, it accepts input text like this (IAST, encoded in UTF-8):
```
pudgaladharmanairātmyayor apratipannavipratipannānām
aviparītapudgaladharmanairātmyapratipādanārthaṃ
triṃśikāvijñaptiprakaraṇārambhaḥ  
pudgaladharmanairātmyapratipādanaṃ punaḥ kleśajñeyāvaraṇaprahāṇārtham  
```

... and produces output like this:

```
pudgala-dharma-nairātmyayoḥ apratipanna-vipratipannānām
aviparīta-pudgala-dharma-nairātmya-pratipādana-artham
triṃśikā-vijñapti-prakaraṇa-ārambhaḥ
pudgala-dharma-nairātmya-pratipādanam punar kleśa-jñeya-āvaraṇa-prahāṇa-artham
```

Please have a look at the paper (in the top directory of this repository) for details about the error rate. We found that it reaches ~15% on the level of text lines; this means that about 85% of all lines processed with the model don't contain wrong Sandhi or compound resolutions.

### Requirements:

The following software must be installed on your machine.
* Python 3.5
* tensorflow (consider to use the latest [Ana-]conda release)
* numpy
* (optional:) h5py, when you want to train a new model


### Steps
On the command line ...
* Change to folder 'code'.
* Run the script apply.py. Make sure to pass the path of the file to be analyzed as the first command line argument to the script. Ex.: `python apply.py c:/my/interesting/text.txt`
* You can pass the output file, in which the analyzed text will be stored, as a second optional parameter. If you don't provide this parameter, the analyzed text will be stored as [path-original].unsandhied.
* On a modern machine, the script should terminate in a few seconds.

### Prerequisites and caveats
* Input is encoded in IAST.
* The input contains only Sanskrit text, but no reference systems, brackets, ...; this means nothing that is not defined in the IAST system.
The analyzer will not crash if it encounters other symbols, but quality will be suboptimal.

Good:
```
omityādeśam ādāya natvā taṃ suravandinaḥ
urvaśīmapsaraḥśreṣṭhāṃ puraskṛtya divaṃ yayuḥ
```

Bad:
```
om it[i-]ādeśam ādāya natvā taṃ sura-vandinaḥ
<br>
urvaśīm apsaraḥ^śreṣṭhāṃ puraskṛtya divaṃ yayuḥ
```

Bad:
```
om ity ādeśam{This is an interesting form!} ādāya natvā [this expression comes from Mbh 22.33.44] taṃ suravandinaḥ // 15.33.7
```
* The pre-trained model has a limit of 128 characters per text line. Longer text lines are cut after the 128th character.
If you need a model for longer text lines, train a new one (see below). 
* The majority of training texts is composed in classical Sanskrit. The model may have problems with (early) Vedic.
* The input sticks to the orthographic conventions used - more or less consistently - in the DCS. If your input text deviates strongly from this convention (e.g., class nasals instead of anusvara), analysis quality may go down.


## Training a new model

### Preparing the data
* Download the training data from [Google Drive](https://drive.google.com/open?id=1Lf1VPxsYRzC3yuYz9XfPbHpYSh7ZOJOO).
* Unzip them into the directory data/input.
* Unzip the files `data/input/sandhi-data-sentences-test.zip` and `data/input/sandhi-data-sentences-validation.zip` into data/input.
* The directory data/input should now contain three files: `sandhi-data-sentences-test.dat`, `sandhi-data-sentences-train.dat` and `sandhi-data-sentences-validation.dat`
* Change to folder 'code'.
* Change the settings in 'configuration.py', if desired. Most important:
  * max_sequence_length_sen: Number of characters per text line
  * max_n_load: How many samples to load. 0 = all samples.
* Run the pre-processing script. Depending on the value of `max_n_load`, this can take quite a lot of time.
```
python preprocess_data.py
```
* The script produces one (huge) hdf5 file and a file for the de-/encoder in data/input. These data are required for training.

### Training
* Change to folder 'code'.
* Change the settings in 'configuration.py', if desired. ***Important***: Make sure that data for the current settings of config.max_sequence_length_sen and config.max_n_load are available in data/input (see *Preparing the data*).
* Run:
```
python train.py
```
* Training on machines without Cuda GPU may take a long time.


### Format of the data files
The first few lines of the test data file look like:

```
# SEN
# TEXT 11 AgRPar
# TOPIC 54 Smrti
$-AratnAni_paQcaDA_uparatnAni_catvAri_kaTayAmi_SRNuta_tat_gomedaM_puzyarAgaM_ca_vExUryaM_ca_pravAlakam
v v NC vajra_67908 v
a a _ _ a
j j _ _ j
r r _ _ r
a a _ _ _
M m _ _ _
```

Notes:
* `# SEN` indicates the start of a new text line.
* `# TEXT 11 AgRPar` gives the unique identifier of the text (11) in the DCS database and its short title (AgRPar); not used in the paper.
* `# TOPIC 54 Smrti` identifier of the topic (54, Smrti); not used in the paper.
* `$-AratnAni_paQcaDA_uparatnAni_ca ...` 100 characters that precede the current line in the text; not used in the paper.
* Lines that consist of five tokens separated by blank spaces (`v v NC vajra_67908 v` or `M m _ _ _`) constitute the actual data. Elements for the line `v v NC vajra_67908 v`:
  * v: original, observed character; input of the classifier (M in the last line of the example)
  * v: "hidden" character to be predicted (m [lowercase] in the last line of the example; anusvara M should be translated into m).
  * NC: POS tag of the underlying word; not used in the paper.
  * vajra_67908: unique identifier of the underlying lexeme; not used in the paper.
  * v: "hidden" character of the unchangeable stem of the lexeme; not used in the paper.
  
  Note that the model in the paper only uses tokens 1 and 2 of each line. Tokens 3-5 were used for internal experiments, are not really validated and may change in future releases.
