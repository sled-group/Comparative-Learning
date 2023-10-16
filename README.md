# Human Inspired Progressive Alignment and Comparative Learning for Grounded Word Acquisition

## TLDR

- **Authors:** Yuwei Bao, Barrett Lattimer, Joyce Chai
- **Organization:** University of Michigan, Computer Science and Engineering
- **Published in:** ACL 2023, Toronto, Canada
- **Links:** [Arxiv](https://arxiv.org/abs/2307.02615), [Github](https://github.com/sled-group/Comparative-Learning/tree/main), [Dataset](https://www.dropbox.com/sh/irnw2jdw3vs9od9/AACB1SqQWeWE7hjJTfhTRhA5a?dl=0)
- :star2: Honorable Mentions for the Best Paper Award




## Abstract
Human language acquisition is an efficient, supervised, and continual process. In this work, we took inspiration from how human babies acquire their first language, and developed a computational process for word acquisition through comparative learning. Motivated by cognitive findings, we generated a small dataset that enables the computation models to compare the similarities and differences of various attributes, learn to filter out and extract the common information for each shared linguistic label. We frame the acquisition of words as not only the information filtration process, but also as representation-symbol mapping. This procedure does not involve a fixed vocabulary size, nor a discriminative objective, and allows the models to continually learn more concepts efficiently. Our results in controlled experiments have shown the potential of this approach for efficient continual learning of grounded words.


## [Dataset] **SOLA**: **S**imulated **O**bjects for **L**anguage **A**cquisition

**SOLA (Simulated Objects for Language Acquisition)** is a small clean dataset with little noise and clearly defined attributes for efficient comparative learning and grounded language acquisition. It is generated using the open\-source simulation software [Kubric](https://github.com/google-research/kubric).

![alt text](https://github.com/sled-group/Comparative-Learning/blob/main/assets/dataset_figure.png)


### Dataset Stats

|Learning Attributes  |Changing Attributes |Variation Attributes |
| ------------- | ------------- | ------------- |
| Color: 8 | Lighting: 3  | Shade: 3 |
| Material: 4  | Camera Angle: 6  | Size: 3 |
| Shape: 11 | |Stretch: 4|
|**Total:**| **6336 (RBGA)** | **989 (RGBA)**|


**Image Types (5):** RGBA, Depth, Surface Normal, Segmentation Map, Object Coordinates


### Dataset Download

- Dropbox Link: [LINK](https://www.dropbox.com/sh/irnw2jdw3vs9od9/AACB1SqQWeWE7hjJTfhTRhA5a?dl=0)
- Hugging Face: [LINK](https://huggingface.co/datasets/sled-umich/SOLA)


## [Method] **Comparative Learning**

Comparative Learning is the process of finding the similarities and differences from a set of inputs. It is a general learning strategy that can be applied to different input modalities, sizes, and duration. It can be broken down to the following two parts:
- **Similarity Learning:** The process of SIM finds similarities across input batches, and extracts out its shared representation 
- **Difference Learning:** The process of DIF highlights the differences between an object label l and other non-compatible labels, and refines the representation for word l

Highlights:
- **Acquisition Process:** We define the word acquisition as two parts of learning: **Information Filteration** and **Representation-Word Mapping**. It is to learn a computation as well as a representation. All learned feature-word mapping will be stored in memory.
- **Continual Learning:** In this work, we compute the centroid of a SIM batch to extract their shared feature, and refine the scope of this feature with the DIF batch. With the help of memory storage, 1) New words can be continually added to the memory; 2) the existing word-feature can be pulled out of the memory, updated and refined when more examples are availble. 

![alt text](https://github.com/sled-group/Comparative-Learning/blob/main/assets/pipeline.png)


### Learning
To train the models from scratch:
```
python main.py [-h] --in_path IN_PATH --out_path OUT_PATH
               [--model_name MODEL_NAME] [--pre_train PRE_TRAIN]

optional arguments:
  -h, --help            show this help message and exit
  --in_path IN_PATH, -i IN_PATH
                        Data input path
  --out_path OUT_PATH, -o OUT_PATH
                        Model memory output path
  --model_name MODEL_NAME, -n MODEL_NAME, default = 'best_mem.pickle'
                        Best model memory to be saved file name
  --pre_train PRE_TRAIN, -p PRE_TRAIN, optional,
                        Pretrained model import name (saved in outpath)
 ```


## What Can the Acquired Models Do?

### Novel Composition Generation
![alt text](https://github.com/sled-group/Comparative-Learning/blob/main/assets/composition.png)


### Composition Reasoning
<!-- ![alt text](https://github.com/sled-group/Comparative-Learning/blob/main/assets/r_c.png)
![alt text](https://github.com/sled-group/Comparative-Learning/blob/main/assets/r_m.png)
![alt text](https://github.com/sled-group/Comparative-Learning/blob/main/assets/r_s.png) -->

![alt text](https://github.com/sled-group/Comparative-Learning/blob/main/assets/reason.png)


## Citation
```
@inproceedings{bao-etal-2023-human,
    title = "Human Inspired Progressive Alignment and Comparative Learning for Grounded Word Acquisition",
    author = "Bao, Yuwei  and
      Lattimer, Barrett  and
      Chai, Joyce",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.863",
    doi = "10.18653/v1/2023.acl-long.863",
    pages = "15475--15493",
    abstract = "Human language acquisition is an efficient, supervised, and continual process. In this work, we took inspiration from how human babies acquire their first language, and developed a computational process for word acquisition through comparative learning. Motivated by cognitive findings, we generated a small dataset that enables the computation models to compare the similarities and differences of various attributes, learn to filter out and extract the common information for each shared linguistic label. We frame the acquisition of words as not only the information filtration process, but also as representation-symbol mapping. This procedure does not involve a fixed vocabulary size, nor a discriminative objective, and allows the models to continually learn more concepts efficiently. Our results in controlled experiments have shown the potential of this approach for efficient continual learning of grounded words.",
}
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. For further questions, please contact yuweibao@umich.edu.


## License
[MIT](https://choosealicense.com/licenses/mit/)

[![DOI](https://zenodo.org/badge/326563198.svg)](https://zenodo.org/badge/latestdoi/326563198)

