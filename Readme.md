---------
MazurkaBL
---------

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/katkost/MazurkaBL/ismir_tutorial)


About the dataset: 
Name identification:

-Chopin Mazurkas: M < opus > - < number >

-PianistID: example "pid1263-01" ->

(from mazurka-discography.txt)

<pre>
opus |key    |performer |year |time |seconds |label                  |pid     |status

6.1  |F# min |Csalog    |1996 |3:16 |196     |Hungaroton HCD 31755/6 |1263-01 |-
</pre>

Interactive plots: https://plot.ly/~katkost/211/?share_key=euAI7btMXU8mlDAuwKFUl4
_________________________________________________________________________________

Folders:
___________
"beat_dyn":
One file per Mazurka.
Columns:

<pre>
1. Number of score bar | 2. Number of score beat in bar | 3-x. Performer dynamic values (normalised sones)
                                                          each column for a recording identified by PianistID

Rows: 1. PianistID | 2-y. Score beats
</pre>
____________
"beat_time":
One file per Mazurka.
Columns:

<pre>
1. Number of score bar | 2. Number of score beat in bar | 3-x. Performer time values (seconds)
                                                          each column for a recording identified by PianistID

Rows: 1. PianistID | 2-y. Score beats
</pre>
___________
"markings": One file per Mazurka.
Rows:
<pre>
1. Expressive markings found in score | 2. Score beat location of marking
</pre>
___________
"sones": One file per Mazurka.

Columns:

<pre>
1. Time (sec.) | 2. sone value (computed using ma_sone algorithm [1])
</pre>

> [1] www.pampalk.at/ma/documentation.html, accessed 3 January 2018.


If using the dataset, please refer to as:

[Katerina Kosta, Oscar F. Bandtlow, and Elaine Chew. "MazurkaBL: Score-aligned loudness, beat, expressive markings data for 2000 Chopin Mazurka recordings”. In Proceedings of the forth International Conference on Technologies for Music Notation and Representation (TENOR). 2018]

Paper: https://drive.google.com/file/d/1h6Ekso91S1U_Ayhx_Y9-NoqV7rxE36ma/view?usp=sharing

Presentation: https://drive.google.com/file/d/1OfrY2jzq_LYt2NnVewM_DVX5ib0-XXBo/view?usp=sharing

____________
Licence: https://creativecommons.org/licenses/by-nc-sa/4.0/
