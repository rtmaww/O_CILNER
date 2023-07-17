# O_CILNER
This is the source code of the ACL 2023 paper: [**Learning 'O' Helps for Learning More: Handling the Unlabeled Entity Problem for Class-incremental NER**](https://aclanthology.org/2023.acl-long.328/)

## Contents

- [Getting Started](#requirements)
  - [Requirements](#requirements)
  - [Dataset](#dataset)
    - [Experiment Setting](#experiment-setting)
    - [Data Format](#data-format)
  - [Structure](#structure)
  - [How to Run](#How-to-Run)
- [Citation](#Citation)

## Requirements

 Run the following script to install the remaining dependencies. Python and CUDA version: `Python 3.7.10 / CUDA 11.7 `

```shell
pip install -r requirements.txt
```

## Dataset

### Experiment Setting
- Few-NERD dataset: 
[Few-NERD dataset](https://ningding97.github.io/fewnerd/), which contains 66 fine-grained entity types. We randomly split the 66 classes in Few-NERD into 11 tasks, corresponding to 11 steps, each of which contains 6 entity classes and an "O" class. The training set and development set of each task contains sentences only labeled with classes of the current task. The test set contains sentences labeled with all learned classes in the task.

The instance data is located in `./data/tasks`.

- OntoNotes 5.0 dataset: 
On the OntoNotes 5.0 dataset, we split 18 classes into 6 tasks in the same way.

### Data Format

The data are pre-processed into the typical NER data forms as below (`token\tlabel`). 

```latex
Between	O
1789	O
and	O
1793	O
he	O
sat	O
on	O
a	O
committee	O
reviewing	O
the	O
administrative	MISC-law
constitution	MISC-law
of	MISC-law
Galicia	MISC-law
to	O
little	O
effect	O
.	O
```

## Structure

The structure of our project is:

```shell
--cil_ner_train
| -- run_incremental_proto.py
| -- run_incremental_rehearsal.py
| -- run_incremental_proto.sh
| -- run_incremental_rehearsal.sh

--data
| -- tasks
| -- labels.txt

--model
| -- supcon_net.py

--util
| -- data_loader.py                 
| -- gather.py    
| -- loss_extendner.py
| -- metric.py
| -- ncm_classifier.py
| -- supervised_util.py        
```

## How to Run

Run `./cil_ner_train/run_incremental_proto.sh` or `./cil_ner_train/run_incremental_rehearsal.sh`.


## Citation

```bibtex
@inproceedings{ma-etal-2023-learning,
    title = "Learning {``}{O}{''} Helps for Learning More: Handling the Unlabeled Entity Problem for Class-incremental {NER}",
    author = "Ma, Ruotian  and
      Chen, Xuanting  and
      Lin, Zhang  and
      Zhou, Xin  and
      Wang, Junzhe  and
      Gui, Tao  and
      Zhang, Qi  and
      Gao, Xiang  and
      Chen, Yun Wen",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.328",
    pages = "5959--5979"
}
```

