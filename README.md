# ERAlign
This is the repository for **ERAlign**, the proposed framework of **"*Unsupervised Robust Cross-Lingual Entity Alignment via Joint Modeling of Entity and Relation Texts*" *(WSDM 2025)***.

Our paper is available at arXiv: [https://arxiv.org/abs/2407.15588](https://arxiv.org/abs/2407.15588)

## Requirement
- python == 3.9
- numpy == 1.24.3
- tensorflow == 2.13.0
- torch == 2.0.1
- tqdm
- sentence-transformers == 2.2.2

## Code
To run with default setting, please use : 
`python main.py`

## Datasets
You can download the dataset [here](https://drive.google.com/file/d/1yeiQmkkPFrcu0Xj6O6nrnBN0i33-bHAG/view?usp=sharing). 



These datasets are provided by:   

*Zequn Sun, Wei Hu, and Chengkai Li. 2017. ***Cross-Lingual Entity Alignment via Joint Attribute-Preserving Embedding***. In The Semantic Web – ISWC 2017*   

## Citation
If our work was helpful for your project, cite our work :)

```bibtex
@inproceedings{10.1145/3701551.3703500,
author = {Yoon, Soojin and Ko, Sungho and Kim, Tongyoung and Kang, SeongKu and Yeo, Jinyoung and Lee, Dongha},
title = {Unsupervised Robust Cross-Lingual Entity Alignment via Neighbor Triple Matching with Entity and Relation Texts},
year = {2025},
isbn = {9798400713293},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3701551.3703500},
doi = {10.1145/3701551.3703500},
booktitle = {Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining},
pages = {184–193},
numpages = {10},
keywords = {cross-lingual entity alignment, knowledge graph, neighbor triple matching, optimal transport, pretrained language models},
location = {Hannover, Germany},
series = {WSDM '25}
}
```
  
