# TT Sounds

We introduct **TT sounds**, a table tennis racket-ball bounce dataset. 

It is used in "Spin Detection using Racket Bounce Sounds in Table tennis" [https://arxiv.org/pdf/2409.11760v1](https://arxiv.org/pdf/2409.11760v1).

## Reproducing results
All the results from the paper can be reproducted with the code in source. 

The dataset needs to be downloaded and put into a data/ directory here.

## Dataset
Distribution of samples across different classes in the dataset. The dataset was subsequently divided into training and testing subsets with an 80/20 split ratio.

| **Surface** | **Back** | **Flat** | **Top** | **Total** |
|-------------|----------|----------|---------|-----------|
| Racket 01   | 263      | 354      | 275     | 892       |
| Racket 02   | 93       | 168      | 40      | 301       |
| Racket 03   | 70       | 162      | 43      | 275       |
| Racket 04   | 98       | 145      | 45      | 288       |
| Racket 05   | 55       | 185      | 0       | 240       |
| Racket 06   | 55       | 152      | 0       | 207       |
| Racket 07   | 60       | 184      | 0       | 244       |
| Racket 08   | 101      | 193      | 42      | 336       |
| Racket 09   | 96       | 159      | 41      | 296       |
| Racket 10   | 100      | 177      | 40      | 317       |
| **Total**   | 991      | 1879     | 526     | 3396      |
| **Table**   |          |          |         | 777       |
| **Floor**   |          |          |         |  290      |
| **Other**   |          |          |         |  1239     |

The dataset can be downloaded with this link: [Nextcloud](https://cloud.cs.uni-tuebingen.de/index.php/s/p3tw3EqE9csXoRn)

The orginal recordings are saved in the *raw_sounds* and the extracted 15ms long dataset samples are saved in *sounds*


## Rackets configuration

| **Blade**     | **Sponge thickness (mm)** | **Rubber**             | **Id** |
|---------------|---------------------------|------------------------|--------|
| **Offensive** | 2.1                       | Inverted (offensive)    | 1      |
| **Offensive** | 1.8                       | Inverted (allround)     | 2      |
| **Defensive** | 2.1                       | Inverted (offensive)    | 3      |
| **Defensive** | 1.8                       | Inverted (allround)     | 4      |
| **Allrounder**| 1.2                       | Long pips               | 5      |
| **Allrounder**| 0                         | Long pips               | 6      |
| **Allrounder**| 1.2                       | Medium pips             | 7      |
| **Allrounder**| 2.0                       | Short pips              | 8      |
| **Allrounder**| 2.1                       | Inverted (Offensive)    | 9      |
| **Allrounder**| 2.1                       | Anti-spin               | 10     |


### Rubbers used

| **Name**                        | **Type**                |
|----------------------------------|-------------------------|
| Tibhar Grass D.TecS              | Long pips               |
| Dr. Neubauer Diamant             | Medium pips             |
| andro Blowfish                   | Short pips              |
| andro Power 3                    | Inverted (allround)      |
| andro Hexer Duro                 | Inverted (offensive)     |
| Dr. Neubauer A-B-S II soft       | Anti-spin               |

### Blades used
| **Name**                         | **Type**                |
|----------------------------------|-------------------------|
| DONIC Appelgren Allplay Senso V1 | Allrounder              |
| Donic Holz Original Carbospeed    | Offensive               |
| Tibhar Holz Defense Plus          | Defensive               |


## License
All files in this dataset are copyright by us and published under the 
Creative Commons Attribution-NonCommerial 4.0 International License, found 
[here](https://creativecommons.org/licenses/by-nc/4.0/).
This means that you must give appropriate credit, provide a link to the license,
and indicate if changes were made. You may do so in any reasonable manner,
but not in any way that suggests the licensor endorses you or your use. You
may not use the material for commercial purposes.
