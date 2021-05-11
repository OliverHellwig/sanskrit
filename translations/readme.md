# Aim

Provide scholarly translations whose sentence numbering runs parallel with that used in the DCS.


# Preparing the data
Regular expressions:

Removing page numbers
\[p\. [0-9]+\] => nil

Removing extra newlines
([^ ])(\r\n){2,20}([a-zA-Z']) => \1 \3

Numbering
^([0-9]+)\.{0,1}  => 1.1.\1 etc.