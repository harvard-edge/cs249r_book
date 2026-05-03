# Bib verification report

- **Checked**: 2026-05-03T14:12:19
- **Total entries**: 1606
- **Verified**: 32
- **Broken**: 99
- **Uncertain**: 2
- **Failed batches**: 33

## Broken (need fixing)

### `vertex_ai_fine_tuning`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Corporate author incorrectly formatted as 'Cloud, Google', which parses as a person's name.
- **Suggested fix**:
  - `author`: {Google Cloud}
  - `title`: Tune models overview
  - `year`: 2024
- **Sources**:
  - https://cloud.google.com/vertex-ai/docs/generative-ai/models/tune-models

### `vertex_ai_model_registry`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Corporate author incorrectly formatted as 'Cloud, Google'.
- **Suggested fix**:
  - `author`: {Google Cloud}
  - `title`: Vertex AI Model Registry
  - `year`: 2024
- **Sources**:
  - https://cloud.google.com/vertex-ai/docs/model-registry/introduction

### `victor2019machine`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 5 mismatch: Key prefix 'victor' uses the author's first name. The surname is 'Sheng'. Subtitle first word missing capitalization protection.
- **Suggested fix**:
  - `author`: Sheng, Victor S. and Zhang, Jing
  - `title`: Machine Learning with Crowdsourcing: {A} Brief Summary of the Past Research and Future Directions
  - `year`: 2019
  - `venue`: Proceedings of the AAAI Conference on Artificial Intelligence
  - `publisher`: Association for the Advancement of Artificial Intelligence (AAAI)
  - `pages`: 9837--9843
  - `rename_key_to`: sheng2019machine
- **Sources**:
  - https://doi.org/10.1609/aaai.v33i01.33019837

### `villalobos2022will`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 1 violation: Acronym 'LLM' is not protected in curly braces.
- **Suggested fix**:
  - `author`: Villalobos, Pablo and Ho, Anson and Sevilla, Jaime and Besiroglu, Tamay and Heim, Lennart and Hobbhahn, Marius
  - `title`: Will we run out of data? Limits of {LLM} scaling based on human-generated data
  - `year`: 2022
- **Sources**:
  - http://arxiv.org/abs/2211.04325v2

### `viola2001rapidobject`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 4a violation: Author initials used instead of full names. Rule 2 violation: CVPR not expanded to official venue string. Rule 3: Publisher needs to be shortened to IEEE.
- **Suggested fix**:
  - `author`: Viola, Paul and Jones, Michael
  - `title`: Rapid object detection using a boosted cascade of simple features
  - `year`: 2001
  - `venue`: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
  - `publisher`: IEEE
  - `pages`: I-511--I-518
- **Sources**:
  - https://doi.org/10.1109/cvpr.2001.990517

### `wachter2017counterfactual`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 1 violation: Subtitle first word 'Automated' and acronym 'GDPR' are not protected in curly braces.
- **Suggested fix**:
  - `author`: Wachter, Sandra and Mittelstadt, Brent and Russell, Chris
  - `title`: Counterfactual Explanations Without Opening the Black Box: {Automated} Decisions and the {GDPR}
  - `year`: 2017
- **Sources**:
  - https://doi.org/10.2139/ssrn.3063289

### `wang2018glue`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 1 violation: Acronym 'GLUE' and subtitle first word 'A' are not protected.
- **Suggested fix**:
  - `author`: Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.
  - `title`: {GLUE}: {A} Multi-Task Benchmark and Analysis Platform for Natural Language Understanding
  - `year`: 2018
- **Sources**:
  - http://arxiv.org/abs/1804.07461v3

### `wang2018skipnet`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 2 violation: ECCV not expanded to official venue string.
- **Suggested fix**:
  - `author`: Wang, Xin and Yu, Fisher and Dou, Zi-Yi and Darrell, Trevor and Gonzalez, Joseph E.
  - `title`: SkipNet: Learning Dynamic Routing in Convolutional Networks
  - `year`: 2018
  - `venue`: European Conference on Computer Vision (ECCV)
  - `publisher`: Springer International Publishing
  - `pages`: 420--436
- **Sources**:
  - https://doi.org/10.1007/978-3-030-01261-8_25

### `wang2019balanced`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 2 violation: ICCV not matching exact official venue string (has year prepended).
- **Suggested fix**:
  - `author`: Wang, Tianlu and Zhao, Jieyu and Yatskar, Mark and Chang, Kai-Wei and Ordonez, Vicente
  - `title`: Balanced Datasets Are Not Enough: Estimating and Mitigating Gender Bias in Deep Image Representations
  - `year`: 2019
  - `venue`: IEEE/CVF International Conference on Computer Vision (ICCV)
  - `publisher`: IEEE
  - `pages`: 5309--5318
- **Sources**:
  - https://doi.org/10.1109/iccv.2019.00541

### `wang2019benchmarking`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 1 violation: Acronyms 'TPU', 'GPU', and 'CPU' are not protected in curly braces.
- **Suggested fix**:
  - `author`: Wang, Yu Emma and Wei, Gu-Yeon and Brooks, David
  - `title`: Benchmarking {TPU}, {GPU}, and {CPU} Platforms for Deep Learning
  - `year`: 2019
- **Sources**:
  - http://arxiv.org/abs/1907.10701v4

### `wang2019superglue`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 1 violation: Acronym 'SuperGLUE' and subtitle first word 'A' are not protected.
- **Suggested fix**:
  - `author`: Wang, Alex and Pruksachatkun, Yada and Nangia, Nikita and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.
  - `title`: {SuperGLUE}: {A} Stickier Benchmark for General-Purpose Language Understanding Systems
  - `year`: 2019
- **Sources**:
  - http://arxiv.org/abs/1905.00537v3

### `warden2018speech`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 1 violation: Subtitle first word 'A' is not protected.
- **Suggested fix**:
  - `author`: Warden, Pete
  - `title`: Speech Commands: {A} Dataset for Limited-Vocabulary Speech Recognition
  - `year`: 2018
- **Sources**:
  - https://doi.org/10.48550/arXiv.1804.03209

### `watson_openscale`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 1 violation: Acronym 'IBM' must be protected.
- **Suggested fix**:
  - `author`: {IBM}
  - `title`: {IBM} Watson OpenScale
  - `year`: 2024
- **Sources**:
  - https://www.ibm.com/cloud/watson-openscale

### `wei2022chain`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 2 violation: NeurIPS not expanded to official venue string.
- **Suggested fix**:
  - `author`: Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and Bosma, Maarten and Ichter, Brian and Xia, Fei and Chi, Ed H. and Le, Quoc V. and Zhou, Denny
  - `title`: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
  - `year`: 2022
  - `venue`: Advances in Neural Information Processing Systems (NeurIPS)
  - `publisher`: Curran Associates
  - `pages`: 24824--24837
- **Sources**:
  - http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html

### `weizenbaum1966eliza`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 1 violation: ELIZA is an acronym/proper noun and should be protected.
- **Suggested fix**:
  - `author`: Weizenbaum, Joseph
  - `title`: {ELIZA}--a computer program for the study of natural language communication between man and machine
  - `year`: 1966
- **Sources**:
  - https://doi.org/10.1145/365153.365168

### `welling2009herding`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 2 and 3 violations: ICML must be fully expanded with acronym and publisher should be PMLR.
- **Suggested fix**:
  - `author`: Welling, Max
  - `title`: Herding dynamical weights to learn
  - `year`: 2009
  - `venue`: International Conference on Machine Learning (ICML)
  - `publisher`: PMLR
  - `pages`: 1121--1128
- **Sources**:
  - https://doi.org/10.1145/1553374.1553517

### `wengert1964simple`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 4a violation: Author initials used instead of full first name.
- **Suggested fix**:
  - `author`: Wengert, Robert E.
  - `title`: A simple automatic derivative evaluation program
  - `year`: 1964
- **Sources**:
  - https://doi.org/10.1145/355586.364791

### `werbos1974beyond`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 1 violation: Subtitle first word 'New' is not protected.
- **Suggested fix**:
  - `author`: Werbos, Paul
  - `title`: Beyond regression: {New} tools for prediction and analysis in the behavioral sciences
  - `year`: 1974
- **Sources**:
  - https://en.wikipedia.org/wiki/Paul_Werbos

### `who2019classification`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Corporate author needs curly braces. Rule 1: ICD should be protected.
- **Suggested fix**:
  - `author`: {World Health Organization}
  - `title`: International Classification of Diseases, 11th Revision ({ICD}-11)
  - `year`: 2019
- **Sources**:
  - https://icd.who.int/en

### `wolpert1997no`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 4a violation: Author initials used instead of full names.
- **Suggested fix**:
  - `author`: Wolpert, David H. and Macready, William G.
  - `title`: No free lunch theorems for optimization
  - `year`: 1997
- **Sources**:
  - https://doi.org/10.1109/4235.585893

### `wu_tensor_2019`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 1 violation: Proper noun 'Tensor' and subtitle first word 'Understanding' are not protected.
- **Suggested fix**:
  - `author`: Wu, Chengfu and Grot, Boris and Hardavellas, Nikos
  - `title`: {Tensor} Cores: {Understanding}, Programming, and Performance Analysis
  - `year`: 2019
- **Sources**:
  - https://doi.org/10.1109/MM.2019.2923951

### `Wu2016`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 2 violation: CVPR not expanded to official venue string.
- **Suggested fix**:
  - `author`: Wu, Jiaxiang and Leng, Cong and Wang, Yuhang and Hu, Qinghao and Cheng, Jian
  - `title`: Quantized Convolutional Neural Networks for Mobile Devices
  - `year`: 2016
  - `venue`: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
  - `publisher`: IEEE
  - `pages`: 4820--4828
- **Sources**:
  - https://doi.org/10.1109/cvpr.2016.521

### `wu2019fbnet`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 2 violation: CVPR not matching exact official venue string (has year prepended).
- **Suggested fix**:
  - `author`: Wu, Bichen and Keutzer, Kurt and Dai, Xiaoliang and Zhang, Peizhao and Wang, Yanghan and Sun, Fei and Wu, Yiming and Tian, Yuandong and Vajda, Peter and Jia, Yangqing
  - `title`: FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search
  - `year`: 2019
  - `venue`: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
  - `publisher`: IEEE
  - `pages`: 10726--10734
- **Sources**:
  - https://doi.org/10.1109/cvpr.2019.01099

### `wu2019machine`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 4a violation: Author list is truncated with an ellipsis symbol rather than listing all authors. Rule 2 violation: HPCA not expanded properly.
- **Suggested fix**:
  - `author`: Wu, Carole-Jean and Brooks, David and Chen, Kevin and Chen, Douglas and Choudhury, Sy and Dukhan, Marat and Hazelwood, Kim and Isaac, Eldad and Jia, Yangqing and Jia, Bill and Leyvand, Tommer and Lu, Hao and Lu, Yang and Qiao, Lin and Reagen, Brandon and Spisak, Joe and Sun, Fei and Tulloch, Andrew and Vajda, Peter and Wang, Xiaodong and Wang, Yanghan and Wasti, Bram and Wu, Yiming and Xian, Ran and Yoo, Sungjoo and Zhang, Peizhao
  - `title`: Machine Learning at Facebook: Understanding Inference at the Edge
  - `year`: 2019
  - `venue`: IEEE International Symposium on High-Performance Computer Architecture (HPCA)
  - `publisher`: IEEE
  - `pages`: 331--344
- **Sources**:
  - https://doi.org/10.1109/hpca.2019.00048

### `wulf1995memory`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 4a violation: First name abbreviated as 'Wm.' instead of 'William'.
- **Suggested fix**:
  - `author`: Wulf, William A. and McKee, Sally A.
  - `title`: Hitting the memory wall
  - `year`: 1995
- **Sources**:
  - https://doi.org/10.1145/216585.216588

### `xiao2022smoothquant`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 5 mismatch: Key year 2022 differs from publication year 2023. Rule 1: Missing curly braces for SmoothQuant and subtitle. Rule 2: Venue ICML must be expanded.
- **Suggested fix**:
  - `author`: Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song
  - `title`: {S}mooth{Q}uant: {Accurate} and Efficient Post-Training Quantization for Large Language Models
  - `year`: 2023
  - `venue`: International Conference on Machine Learning (ICML)
  - `publisher`: PMLR
  - `pages`: 38087--38099
  - `rename_key_to`: xiao2023smoothquant
- **Sources**:
  - https://proceedings.mlr.press/v202/xiao23c.html

### `xin-etal-2021-berxit`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 1 violation: Acronyms 'BERxiT', 'BERT' and subtitle first word 'Early' are not protected.
- **Suggested fix**:
  - `author`: Xin, Ji and Tang, Raphael and Yu, Yaoliang and Lin, Jimmy
  - `title`: {BERxiT}: {Early} Exiting for {BERT} with Better Fine-Tuning and Extension to Regression
  - `year`: 2021
- **Sources**:
  - https://doi.org/10.18653/v1/2021.eacl-main.8

### `xinyu`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 5 mismatch: Key does not match the first author's surname (Gholami) or year (2021).
- **Suggested fix**:
  - `author`: Gholami, Amir and Kim, Sehoon and Dong, Zhen and Yao, Zhewei and Mahoney, Michael W. and Keutzer, Kurt
  - `title`: A Survey of Quantization Methods for Efficient Neural Network Inference
  - `year`: 2021
  - `rename_key_to`: gholami2021survey
- **Sources**:
  - https://arxiv.org/abs/2103.13630

### `xla2020`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 1 violation: Corporate author Google should be in braces.
- **Suggested fix**:
  - `author`: {Google}
  - `title`: {XLA}: Optimizing Compiler for Machine Learning
  - `year`: 2020
- **Sources**:
  - https://www.tensorflow.org/xla

### `xu2018alternating`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 2 violation: ICLR must be fully expanded.
- **Suggested fix**:
  - `author`: Xu, Chen and Yao, Jianqiang and Lin, Zhouchen and Ou, Wenwu and Cao, Yuanbin and Wang, Zhirong and Zha, Hongbin
  - `title`: Alternating Multi-bit Quantization for Recurrent Neural Networks
  - `year`: 2018
  - `venue`: International Conference on Learning Representations (ICLR)
  - `publisher`: OpenReview.net
- **Sources**:
  - https://openreview.net/forum?id=S19dR9x0b

### `yang2020coexploration`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Silent corruption: The DOI 10.1002/9783527667703.ch67, publisher, and pages belong to a 2013 Wiley book chapter on Memristive Systems, not the Yang 2020 paper. The Yang paper was an arXiv preprint/DAC 2020 publication.
- **Suggested fix**:
  - `author`: Yang, Lei and Yan, Zheyu and Li, Meng and Kwon, Hyoukjun and Lai, Liangzhen and Krishna, Tushar and Chandra, Vikas and Jiang, Weiwen and Shi, Yiyu
  - `title`: Co-Exploration of Neural Architectures and Heterogeneous ASIC Accelerator Designs Targeting Multiple Tasks
  - `year`: 2020
  - `venue`: arXiv preprint arXiv:2002.04116
- **Sources**:
  - http://arxiv.org/abs/2002.04116v1

### `yang2020resolution`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 2 violation: CVPR must be properly expanded.
- **Suggested fix**:
  - `author`: Yang, Le and Han, Yizeng and Chen, Xi and Song, Shiji and Dai, Jifeng and Huang, Gao
  - `title`: Resolution Adaptive Networks for Efficient Inference
  - `year`: 2020
  - `venue`: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
  - `publisher`: IEEE
  - `pages`: 2366--2375
- **Sources**:
  - https://doi.org/10.1109/cvpr42600.2020.00244

### `yao2021hawq`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 2 violation: ICML must be properly expanded.
- **Suggested fix**:
  - `author`: Yao, Zhewei and Gholami, Amir and Shen, Sheng and Keutzer, Kurt and Mahoney, Michael W.
  - `title`: HAWQ-V3: Dyadic Neural Network Quantization
  - `year`: 2021
  - `venue`: International Conference on Machine Learning (ICML)
  - `publisher`: PMLR
  - `pages`: 11875--11886
- **Sources**:
  - https://proceedings.mlr.press/v139/yao21a.html

### `yosinski2014transferable`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 4: Missing page numbers. Rule 2: NeurIPS must be expanded. Rule 3: Missing publisher.
- **Suggested fix**:
  - `author`: Yosinski, Jason and Clune, Jeff and Bengio, Yoshua and Lipson, Hod
  - `title`: How transferable are features in deep neural networks?
  - `year`: 2014
  - `venue`: Advances in Neural Information Processing Systems (NeurIPS)
  - `publisher`: Curran Associates
  - `pages`: 3320--3328
- **Sources**:
  - https://proceedings.neurips.cc/paper/2014/hash/3b8a614226a953a8cd9526fca6fe9ba5-Abstract.html

### `you2019scaling`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 4: Missing page numbers. Rule 2: MLSys must be expanded.
- **Suggested fix**:
  - `author`: You, Yang and Zhang, Zhao and Hsieh, Cho-Jui and Demmel, James and Keutzer, Kurt
  - `title`: Scaling SGD Batch Size to 32K for ImageNet Training
  - `year`: 2019
  - `venue`: Proceedings of Machine Learning and Systems (MLSys)
  - `pages`: 1--12
- **Sources**:
  - https://proceedings.mlsys.org/paper_files/paper/2019/hash/1b73e8e2fa1aeb5b31d2ba13a3036aaa-Abstract.html

### `yu2022orca`  (book/quarto/contents/vol1/backmatter/references.bib)

- **Issue**: Rule 2 violation: OSDI must be officially expanded.
- **Suggested fix**:
  - `author`: Yu, Gyeong-In and Jeong, Joo Seong and Kim, Geon-Woo and Kim, Soojeong and Chun, Byung-Gon
  - `title`: Orca: A Distributed Serving System for Transformer-Based Generative Models
  - `year`: 2022
  - `venue`: USENIX Symposium on Operating Systems Design and Implementation (OSDI)
  - `publisher`: USENIX Association
  - `pages`: 521--538
- **Sources**:
  - https://www.usenix.org/conference/osdi22/presentation/yu

### `xiao2023smoothquant`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Title acronyms and proper nouns lack braces. Author names include DBLP artifacts (0002, 0003). Venue must be expanded to full name.
- **Suggested fix**:
  - `title`: {SmoothQuant}: Accurate and Efficient Post-Training Quantization for {Large Language Models}
  - `author`: Xiao, Guangxuan and Lin, Ji and Seznec, Micka\"el and Wu, Hao and Demouth, Julien and Han, Song
  - `year`: 2023
  - `venue`: Proceedings of the 40th International Conference on Machine Learning (ICML)
  - `publisher`: PMLR
  - `pages`: 38087--38099
- **Sources**:
  - https://proceedings.mlr.press/v202/xiao23c.html

### `xu2017feature`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Year is 2018 (NDSS 2018), which requires a key rename to match the year token rule.
- **Suggested fix**:
  - `title`: Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks
  - `author`: Xu, Weilin and Evans, David and Qi, Yanjun
  - `year`: 2018
  - `venue`: Proceedings of the Network and Distributed System Security Symposium (NDSS)
  - `rename_key_to`: xu2018feature
- **Sources**:
  - https://www.ndss-symposium.org/ndss2018/submissions/feature-squeezing-detecting-adversarial-examples-deep-neural-networks/

### `Xu2021edge`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Journal abbreviation needs braces for proper nouns/acronyms.
- **Suggested fix**:
  - `title`: Edge Intelligence: Architectures, Challenges, and Applications
  - `author`: Xu, Xiaolong and Li, Fan and Zhang, Wei and He, Liang and Li, Ruidong
  - `year`: 2021
  - `venue`: {IEEE} Internet of Things Journal
  - `pages`: 4229--4249
- **Sources**:
  - https://ieeexplore.ieee.org/document/9320525

### `yao1982protocols`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Venue must be expanded to the official full name with acronym at the end (FOCS).
- **Suggested fix**:
  - `title`: Protocols for secure computations
  - `author`: Yao, Andrew C.
  - `year`: 1982
  - `venue`: 23rd Annual Symposium on Foundations of Computer Science (FOCS)
  - `publisher`: IEEE
  - `pages`: 160--164
- **Sources**:
  - https://doi.org/10.1109/sfcs.1982.38

### `yeh1996triple`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: IEEE needs braces in the venue name.
- **Suggested fix**:
  - `title`: Triple-triple redundant 777 primary flight computer
  - `author`: Yeh, Y.C.
  - `year`: 1996
  - `venue`: 1996 {IEEE} Aerospace Applications Conference. Proceedings
  - `publisher`: IEEE
  - `pages`: 293--307
- **Sources**:
  - https://doi.org/10.1109/aero.1996.495891

### `yin2018byzantine`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Venue must be expanded with the ICML acronym at the end.
- **Suggested fix**:
  - `title`: Byzantine-robust distributed learning: Towards optimal statistical rates
  - `author`: Yin, Dong and Chen, Yudong and Ramchandran, Kannan and Bartlett, Peter
  - `year`: 2018
  - `venue`: International Conference on Machine Learning (ICML)
  - `publisher`: PMLR
  - `pages`: 5650--5659
- **Sources**:
  - https://proceedings.mlr.press/v80/yin18a.html

### `yoo2003slurm`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Title acronyms/proper nouns (SLURM, Linux) lack braces. Publisher field contains location (Berlin Heidelberg) which must be removed.
- **Suggested fix**:
  - `title`: {SLURM}: Simple {Linux} Utility for Resource Management
  - `author`: Yoo, Andy B. and Jette, Morris A. and Grondona, Mark
  - `year`: 2003
  - `venue`: Job Scheduling Strategies for Parallel Processing
  - `publisher`: Springer
  - `pages`: 44--60
- **Sources**:
  - https://doi.org/10.1007/10968987_3

### `you2020large`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Title acronym (BERT) lacks braces. Venue must be expanded with ICLR acronym.
- **Suggested fix**:
  - `title`: Large Batch Optimization for Deep Learning: Training {BERT} in 76 minutes
  - `author`: You, Yang and Li, Jing and Reddi, Sashank and Hseu, Jonathan and Kumar, Sanjiv and Bhojanapalli, Srinadh and Song, Xiaodan and Demmel, James and Keutzer, Kurt and Hsieh, Cho-Jui
  - `year`: 2020
  - `venue`: International Conference on Learning Representations (ICLR)
  - `publisher`: OpenReview.net
- **Sources**:
  - https://openreview.net/forum?id=Syx4wnEtvH

### `young1974first`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Journal abbreviation needs braces for ACM.
- **Suggested fix**:
  - `title`: A first order approximation to the optimum checkpoint interval
  - `author`: Young, John W.
  - `year`: 1974
  - `venue`: Communications of the {ACM}
  - `publisher`: Association for Computing Machinery (ACM)
  - `pages`: 530--531
- **Sources**:
  - https://doi.org/10.1145/361147.361115

### `yu2022orca`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Title proper nouns lack braces. Venue must be expanded to OSDI.
- **Suggested fix**:
  - `title`: {Orca}: A Distributed Serving System for {Transformer}-Based Generative Models
  - `author`: Yu, Gyeong-In and Jeong, Joo Seong and Kim, Geon-Woo and Kim, Soojeong and Chun, Byung-Gon
  - `year`: 2022
  - `venue`: 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI)
  - `publisher`: USENIX Association
  - `pages`: 521--538
- **Sources**:
  - https://www.usenix.org/conference/osdi22/presentation/yu

### `yurdakul2020statistical`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Missing page numbers for journal article.
- **Suggested fix**:
  - `title`: Statistical properties of the population stability index
  - `author`: Yurdakul, Bilal and Naranjo, Joshua
  - `year`: 2020
  - `venue`: The Journal of Risk Model Validation
  - `publisher`: Infopro Digital Services Limited
  - `pages`: 89--100
- **Sources**:
  - https://doi.org/10.21314/jrmv.2020.227

### `zafrir2019q8bert`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Title acronyms lack braces. Missing NeurIPS acronym in venue.
- **Suggested fix**:
  - `title`: {Q8BERT}: Quantized 8Bit {BERT}
  - `author`: Zafrir, Ofir and Boudoukh, Guy and Izsak, Peter and Wasserblat, Moshe
  - `year`: 2019
  - `venue`: 2019 Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing - NeurIPS Edition (EMC2-NIPS)
  - `publisher`: IEEE
  - `pages`: 36--39
- **Sources**:
  - https://doi.org/10.1109/emc2-nips53020.2019.00016

### `zaharia2012resilient`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Venue must be expanded to full name.
- **Suggested fix**:
  - `title`: Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing
  - `author`: Zaharia, Matei and Chowdhury, Mosharaf and Das, Tathagata and Dave, Ankur and Ma, Justin and McCauley, Murphy and Franklin, Michael J and Shenker, Scott and Stoica, Ion
  - `year`: 2012
  - `venue`: Proceedings of the 9th USENIX Symposium on Networked Systems Design and Implementation (NSDI)
  - `publisher`: USENIX Association
  - `pages`: 15--28
- **Sources**:
  - https://www.usenix.org/conference/nsdi12/technical-sessions/presentation/zaharia

### `zaharia2016apache`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Title proper nouns lack braces. ACM in journal needs braces.
- **Suggested fix**:
  - `title`: {Apache Spark}
  - `author`: Zaharia, Matei and Xin, Reynold S. and Wendell, Patrick and Das, Tathagata and Armbrust, Michael and Dave, Ankur and Meng, Xiangrui and Rosen, Josh and Venkataraman, Shivaram and Franklin, Michael J. and Ghodsi, Ali and Gonzalez, Joseph and Shenker, Scott and Stoica, Ion
  - `year`: 2016
  - `venue`: Communications of the {ACM}
  - `publisher`: Association for Computing Machinery (ACM)
  - `pages`: 56--65
- **Sources**:
  - https://doi.org/10.1145/2934664

### `zaharia2018accelerating`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: IEEE needs braces in journal.
- **Suggested fix**:
  - `title`: Accelerating the Machine Learning Lifecycle with {MLflow}
  - `author`: Zaharia, Matei and Chen, Andrew and Davidson, Aaron and Ghodsi, Ali and Hong, Sue Ann and Konwinski, Andy and Murching, Siddharth and Nykodym, Tomas and Ogilvie, Paul and Parkhe, Mani and Xie, Fen and Zumar, Corey
  - `year`: 2018
  - `venue`: {IEEE} Data Engineering Bulletin
  - `publisher`: IEEE
  - `pages`: 39--45
- **Sources**:
  - http://sites.computer.org/debull/A18dec/p39.pdf

### `zhang2008distribution`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Author name should be in Last, First format. IEEE needs braces.
- **Suggested fix**:
  - `title`: On the Distribution of Software Faults
  - `author`: Zhang, Hongyu
  - `year`: 2008
  - `venue`: {IEEE} Transactions on Software Engineering
  - `publisher`: Institute of Electrical and Electronics Engineers (IEEE)
  - `pages`: 301--302
- **Sources**:
  - https://doi.org/10.1109/tse.2007.70771

### `zhang2016understanding`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Year mismatch (2017 paper but key suggests 2016) requiring key rename. Venue needs ICLR acronym.
- **Suggested fix**:
  - `title`: Understanding deep learning requires rethinking generalization
  - `author`: Zhang, Chiyuan and Bengio, Samy and Hardt, Moritz and Recht, Benjamin and Vinyals, Oriol
  - `year`: 2017
  - `venue`: International Conference on Learning Representations (ICLR)
  - `publisher`: OpenReview.net
  - `rename_key_to`: zhang2017understanding
- **Sources**:
  - https://openreview.net/forum?id=Sy8gdB9xx

### `zhang2018analyzing`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: IEEE needs braces in venue.
- **Suggested fix**:
  - `title`: Analyzing and mitigating the impact of permanent faults on a systolic array based neural network accelerator
  - `author`: Zhang, Jeff Jun and Gu, Tianyu and Basu, Kanad and Garg, Siddharth
  - `year`: 2018
  - `venue`: 2018 {IEEE} 36th VLSI Test Symposium (VTS)
  - `publisher`: IEEE
  - `pages`: 1--6
- **Sources**:
  - https://doi.org/10.1109/vts.2018.8368656

### `zhang2018review`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: CSEE needs braces in journal.
- **Suggested fix**:
  - `title`: Review on the research and practice of deep learning and reinforcement learning in smart grids
  - `author`: Zhang, Dongxia and Han, Xiaoqing and Deng, Chunyu
  - `year`: 2018
  - `venue`: {CSEE} Journal of Power and Energy Systems
  - `publisher`: Power System Technology Press
  - `pages`: 362--370
- **Sources**:
  - https://doi.org/10.17775/cseejpes.2018.00520

### `zhang2018thundervolt`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Title proper nouns lack braces. ACM/ESDA/IEEE need braces.
- **Suggested fix**:
  - `title`: {ThUnderVolt}: Enabling Aggressive Voltage Underscaling and Timing Error Resilience for Energy Efficient Deep Learning Accelerators
  - `author`: Zhang, Jeff and Rangineni, Kartheek and Ghodsi, Zahra and Garg, Siddharth
  - `year`: 2018
  - `venue`: 2018 55th {ACM}/{ESDA}/{IEEE} Design Automation Conference (DAC)
  - `publisher`: IEEE
  - `pages`: 1--6
- **Sources**:
  - https://doi.org/10.1109/dac.2018.8465918

### `zhao2018fpga`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: FPGA and IEEE need braces.
- **Suggested fix**:
  - `title`: {FPGA}-Based Remote Power Side-Channel Attacks
  - `author`: Zhao, Mark and Suh, G. Edward
  - `year`: 2018
  - `venue`: 2018 {IEEE} Symposium on Security and Privacy (SP)
  - `publisher`: IEEE
  - `pages`: 229--244
- **Sources**:
  - https://doi.org/10.1109/sp.2018.00049

### `zhao2023fsdp`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Incomplete author list (ends in 'and…'). Title proper noun lacks braces.
- **Suggested fix**:
  - `title`: {PyTorch} {FSDP}: Experiences on Scaling Fully Sharded Data Parallel
  - `author`: Zhao, Yanli and Gu, Andrew and Varma, Rohan and Luo, Liang and Huang, Chien-Chin and Xu, Min and Wright, Less and Shojanazeri, Hamid and Ott, Myle and Shleifer, Sam and Desmaison, Alban and Balioglu, Can and Damania, Pritam and Nguyen, Bernard and Chauhan, Geeta and Hao, Yuchen and Mathews, Ajit and Li, Shen
  - `year`: 2023
  - `venue`: Proceedings of the VLDB Endowment
  - `publisher`: Association for Computing Machinery (ACM)
  - `pages`: 3848--3860
- **Sources**:
  - https://doi.org/10.14778/3611540.3611569

### `zhao2023pytorch`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Incomplete author list (ends in 'and…'). Title proper noun lacks braces. Also a duplicate of zhao2023fsdp.
- **Suggested fix**:
  - `title`: {PyTorch} {FSDP}: Experiences on Scaling Fully Sharded Data Parallel
  - `author`: Zhao, Yanli and Gu, Andrew and Varma, Rohan and Luo, Liang and Huang, Chien-Chin and Xu, Min and Wright, Less and Shojanazeri, Hamid and Ott, Myle and Shleifer, Sam and Desmaison, Alban and Balioglu, Can and Damania, Pritam and Nguyen, Bernard and Chauhan, Geeta and Hao, Yuchen and Mathews, Ajit and Li, Shen
  - `year`: 2023
  - `venue`: Proceedings of the VLDB Endowment
  - `publisher`: Association for Computing Machinery (ACM)
  - `pages`: 3848--3860
- **Sources**:
  - https://doi.org/10.14778/3611540.3611569

### `zhou2018learning`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Venue must be expanded to include CVPR acronym at the end.
- **Suggested fix**:
  - `title`: Learning Rich Features for Image Manipulation Detection
  - `author`: Zhou, Peng and Han, Xintong and Morariu, Vlad I. and Davis, Larry S.
  - `year`: 2018
  - `venue`: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
  - `publisher`: IEEE
  - `pages`: 1053--1061
- **Sources**:
  - https://doi.org/10.1109/cvpr.2018.00116

### `zhu2015dcqcn`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: ACM/SIGCOMM and RDMA lack braces. Venue needs expansion.
- **Suggested fix**:
  - `title`: Congestion Control for Large-Scale {RDMA} Deployments
  - `author`: Zhu, Yibo and Eran, Haggai and Firestone, Daniel and Guo, Chuanxiong and Lipshteyn, Marina and Liron, Yehonatan and Padhye, Jitendra and Raindel, Shachar and Yahia, Mohamad Haj and Zhang, Ming
  - `year`: 2015
  - `venue`: {ACM} {SIGCOMM} Computer Communication Review
  - `publisher`: Association for Computing Machinery (ACM)
  - `pages`: 523--536
- **Sources**:
  - https://doi.org/10.1145/2829988.2787484

### `zhu2019deep`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Venue lacks NeurIPS acronym. Publisher incorrect format.
- **Suggested fix**:
  - `title`: Deep Leakage from Gradients
  - `author`: Zhu, Ligeng and Liu, Zhijian and Han, Song
  - `year`: 2019
  - `venue`: Advances in Neural Information Processing Systems (NeurIPS)
  - `publisher`: Curran Associates
  - `pages`: 14747--14756
- **Sources**:
  - https://papers.nips.cc/paper/9617-deep-leakage-from-gradients

### `ziegler1996ibm`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Author list is incomplete and ends in 'Sull…'. Title needs braces for IBM.
- **Suggested fix**:
  - `title`: {IBM} experiments in soft fails in computer electronics (1978--1994)
  - `author`: Ziegler, J. F. and Curtis, H. W. and Muhlfeld, H. P. and Montrose, C. J. and Chin, B. and Nicewicz, M. and Russell, C. A. and Wang, W. Y. and Freeman, L. B. and Hosier, P. and LaFave, L. E. and Walsh, J. L. and Orro, J. M. and Unger, G. J. and Ross, J. M. and O'Gorman, T. J. and Messina, B. and Sullivan, T. D. and Sykes, A. J. and Yourke, H. and Enger, T. A. and Tolat, V. and Scott, T. S. and Taber, A. H. and Sussman, R. J. and Klein, W. A. and Wahaus, C. W.
  - `year`: 1996
  - `venue`: {IBM} Journal of Research and Development
  - `publisher`: IBM
  - `pages`: 3--18
- **Sources**:
  - https://doi.org/10.1147/rd.401.0003

### `zoph2016neural`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Year mismatch (2017 paper but key suggests 2016) requiring key rename.
- **Suggested fix**:
  - `title`: Neural Architecture Search with Reinforcement Learning
  - `author`: Zoph, Barret and Le, Quoc V.
  - `year`: 2017
  - `venue`: International Conference on Learning Representations (ICLR)
  - `publisher`: OpenReview.net
  - `rename_key_to`: zoph2017neural
- **Sources**:
  - https://arxiv.org/abs/1611.01578

### `zoph2017neural`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Year mismatch (2016 preprint but key suggests 2017) requiring key rename.
- **Suggested fix**:
  - `title`: Neural Architecture Search with Reinforcement Learning
  - `author`: Zoph, Barret and Le, Quoc V.
  - `year`: 2016
  - `venue`: International Conference on Learning Representations (ICLR)
  - `rename_key_to`: zoph2016neural
- **Sources**:
  - http://arxiv.org/abs/1611.01578v2

### `zu2024tpuv4`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Title needs braces for Google and TPUv4. Venue needs acronym.
- **Suggested fix**:
  - `title`: Resiliency at Scale: Managing {Google}'s {TPUv4} Machine Learning Supercomputer
  - `author`: Zu, Yazhou and Ghaffarkhah, Alireza and Dang, Hoang-Vu and Towles, Brian and Hand, Steven and Huda, Safeen and Bello, Adekunle and Kolbasov, Alexander and Rezaei, Arash and Du, Dayou and Lacy, Steve and Wang, Hang and Wisner, Aaron and Lewis, Chris and Bahini, Henri
  - `year`: 2024
  - `venue`: 21st USENIX Symposium on Networked Systems Design and Implementation (NSDI)
  - `publisher`: USENIX Association
  - `pages`: 761--774
- **Sources**:
  - https://www.usenix.org/conference/nsdi24/presentation/zu

### `chandola2009anomaly`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Title is missing its subtitle. Subtitle must be protected and capitalized.
- **Suggested fix**:
  - `title`: Anomaly detection: {A} survey
  - `author`: Varun Chandola and Arindam Banerjee and Vipin Kumar
  - `year`: 2009
  - `venue`: ACM Computing Surveys
  - `publisher`: Association for Computing Machinery (ACM)
  - `pages`: 1--58
  - `doi`: 10.1145/1541880.1541882
- **Sources**:
  - https://dl.acm.org/doi/10.1145/1541880.1541882

### `chang2008bigtable`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Title is missing its subtitle. Proper noun Bigtable must be protected.
- **Suggested fix**:
  - `title`: {Bigtable}: {A} Distributed Storage System for Structured Data
  - `author`: Fay Chang and Jeffrey Dean and Sanjay Ghemawat and Wilson C. Hsieh and Deborah A. Wallach and Mike Burrows and Tushar Chandra and Andrew Fikes and Robert E. Gruber
  - `year`: 2008
  - `venue`: ACM Transactions on Computer Systems
  - `publisher`: Association for Computing Machinery (ACM)
  - `pages`: 1--26
  - `doi`: 10.1145/1365815.1365816
- **Sources**:
  - https://dl.acm.org/doi/10.1145/1365815.1365816

### `chen2019closer`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: ICLR venue must be expanded per rules. Publisher OpenReview.net is missing.
- **Suggested fix**:
  - `title`: A closer look at few-shot classification
  - `author`: Wei-Yu Chen and Yen-Cheng Liu and Zsolt Kira and Yu-Chiang Frank Wang and Jia-Bin Huang
  - `year`: 2019
  - `venue`: International Conference on Learning Representations (ICLR)
  - `publisher`: OpenReview.net
- **Sources**:
  - https://openreview.net/forum?id=HkxLXnAcFQ

### `chen2019looks`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Subtitle requires protection. Venue NeurIPS must be expanded. Publisher must be Curran Associates.
- **Suggested fix**:
  - `title`: This Looks Like That: {Deep} Learning for Interpretable Image Recognition
  - `author`: Chaofan Chen and Oscar Li and Daniel Tao and Alina Barnett and Cynthia Rudin and Jonathan Su
  - `year`: 2019
  - `venue`: Advances in Neural Information Processing Systems (NeurIPS)
  - `publisher`: Curran Associates, Inc.
  - `pages`: 8928--8939
- **Sources**:
  - https://proceedings.neurips.cc/paper/2019/hash/adf7ee2dcf142b0e11888e72b43fcb75-Abstract.html

### `chen2019sc`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Title is incomplete and contains HTML tags instead of braces. Venue needs expansion.
- **Suggested fix**:
  - `title`: {BinFI}: {An} Efficient Fault Injector for Safety-Critical Machine Learning Systems
  - `author`: Zitao Chen and Guanpeng Li and Karthik Pattabiraman and Nathan DeBardeleben
  - `year`: 2019
  - `venue`: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC)
  - `publisher`: ACM
  - `pages`: 1--23
  - `doi`: 10.1145/3295500.3356177
- **Sources**:
  - https://dl.acm.org/doi/10.1145/3295500.3356177

### `chen2020simclr`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Venue missing ICML expansion. Publisher should be PMLR.
- **Suggested fix**:
  - `title`: A Simple Framework for Contrastive Learning of Visual Representations
  - `author`: Ting Chen and Simon Kornblith and Mohammad Norouzi and Geoffrey Hinton
  - `year`: 2020
  - `venue`: Proceedings of the 37th International Conference on Machine Learning (ICML)
  - `publisher`: PMLR
  - `pages`: 1597--1607
  - `doi`: 10.48550/arXiv.2002.05709
- **Sources**:
  - http://proceedings.mlr.press/v119/chen20j.html

### `chen2020tensorfi`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Proper nouns TensorFI and TensorFlow must be protected. Subtitle requires protection.
- **Suggested fix**:
  - `title`: {TensorFI}: {A} Flexible Fault Injection Framework for {TensorFlow} Applications
  - `author`: Zitao Chen and Niranjhana Narayanan and Bo Fang and Guanpeng Li and Karthik Pattabiraman and Nathan DeBardeleben
  - `year`: 2020
  - `venue`: 2020 IEEE 31st International Symposium on Software Reliability Engineering (ISSRE)
  - `publisher`: IEEE
  - `pages`: 426--435
  - `doi`: 10.1109/issre5003.2020.00047
- **Sources**:
  - https://ieeexplore.ieee.org/document/9251052

### `cheng2016clear`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Title is incomplete. Acronym CLEAR must be protected.
- **Suggested fix**:
  - `title`: {CLEAR}: {Cross-Layer} Exploration for Architecting Resilience
  - `author`: Eric Cheng and Shahrzad Mirkhani and Lukasz G. Szafaryn and Chen-Yong Cher and Hyungmin Cho and Kevin Skadron and Mircea R. Stan and Klas Lilja and Jacob A. Abraham and Pradip Bose and Subhasish Mitra
  - `year`: 2016
  - `venue`: Proceedings of the 53rd Annual Design Automation Conference
  - `publisher`: ACM
  - `pages`: 1--6
  - `doi`: 10.1145/2897937.2897996
- **Sources**:
  - https://dl.acm.org/doi/10.1145/2897937.2897996

### `chollet2017xception`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Proper noun Xception must be protected. Subtitle requires protection. Venue CVPR must be expanded per rules.
- **Suggested fix**:
  - `title`: {Xception}: {Deep} Learning with Depthwise Separable Convolutions
  - `author`: Francois Chollet
  - `year`: 2017
  - `venue`: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
  - `publisher`: IEEE
  - `pages`: 1800--1807
  - `doi`: 10.1109/cvpr.2017.195
- **Sources**:
  - https://ieeexplore.ieee.org/document/8099678

### `chouldechova2017fair`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Subtitle requires protection.
- **Suggested fix**:
  - `title`: Fair Prediction with Disparate Impact: {A} Study of Bias in Recidivism Prediction Instruments
  - `author`: Alexandra Chouldechova
  - `year`: 2017
  - `venue`: Big Data
  - `publisher`: Mary Ann Liebert Inc
  - `pages`: 153--163
  - `doi`: 10.1089/big.2016.0047
- **Sources**:
  - https://www.liebertpub.com/doi/10.1089/big.2016.0047

### `chowdhery2022palm`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Author list is truncated. Acronym PaLM must be protected. Subtitle requires protection.
- **Suggested fix**:
  - `title`: {PaLM}: {Scaling} Language Modeling with Pathways
  - `author`: Aakanksha Chowdhery and Sharan Narang and Jacob Devlin and Maarten Bosma and Gaurav Mishra and Adam Roberts and Paul Barham and Hyung Won Chung and Charles Sutton and Sebastian Gehrmann and Parker Schuh and Kensen Shi and Sasha Tsvyashchenko and Joshua Maynez and Abhishek Rao and Parker Barnes and Yi Tay and Noam Shazeer and Vinodkumar Prabhakaran and Emily Reif and Kevin Du and Ben Hutchinson and Reiner Pope and James Bradbury and Jacob Austin and Michael Isard and Guy Gur-Ari and Pengcheng Yin and Toju Duke and Anselm Levskaya and Sanjay Ghemawat and Sunipa Dev and Henryk Michalewski and Xavier Garcia and Vedant Misra and Kevin Robinson and Liam Fedus and Denny Zhou and Daphne Ippolito and David Luan and Hyeontaek Lim and Barret Zoph and Alexander Spiridonov and Ryan Sepassi and David Dohan and Shivani Agrawal and Mark Omernick and Andrew M. Dai and Thanumalayan Sankaranarayana Pillai and Marie Pellat and Aitor Lewkowycz and Erica Moreira and Rewon Child and Oleksandr Polozov and Katherine Lee and Zongwei Zhou and Xuezhi Wang and Brennan Saeta and Mark Diaz and Orhan Firat and Michele Catasta and Jason Wei and Kathy Meier-Hellstern and Douglas Eck and Jeff Dean and Slav Petrov and Noah Fiedel
  - `year`: 2022
  - `venue`: arXiv preprint arXiv:2204.02311
- **Sources**:
  - https://arxiv.org/abs/2204.02311

### `christiano2017deep`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Venue NeurIPS must be expanded per rules.
- **Suggested fix**:
  - `title`: Deep Reinforcement Learning from Human Preferences
  - `author`: Paul F. Christiano and Jan Leike and Tom B. Brown and Miljan Martic and Shane Legg and Dario Amodei
  - `year`: 2017
  - `venue`: Advances in Neural Information Processing Systems (NeurIPS)
  - `publisher`: Curran Associates, Inc.
  - `pages`: 4299--4307
- **Sources**:
  - https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html

### `claude2022constitutional`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Key mismatch with author field (Anthropic) and actual paper authors. Acronym AI must be protected. Subtitle requires protection.
- **Suggested fix**:
  - `title`: Constitutional {AI}: {Harmlessness} from {AI} Feedback
  - `author`: Yuntao Bai and Saurav Kadavath and Sandipan Kundu and Amanda Askell and Jackson Kernion and Andy Jones and Anna Chen and Anna Goldie and Azalia Mirhoseini and Cameron McKinnon and Carol Chen and Catherine Olsson and Christopher Olah and Danny Hernandez and Dawn Drain and Deep Ganguli and Dustin Li and Eli Tran-Johnson and Ethan Perez and Jamie Kerr and Jared Mueller and Jeffrey Ladish and Joshua Landau and Kamal Ndousse and Kamile Lukosuite and Liane Lovitt and Michael Sellitto and Nelson Elhage and Nicholas Schiefer and Noemi Mercado and Nova DasSarma and Robert Lasenby and Robin Larson and Sam Ringer and Scott Johnston and Shauna Kravec and Sheer El Showk and Stanislav Fort and Tamera Lanham and Timothy Telleen-Lawton and Tom Conerly and Tom Henighan and Tristan Hume and Samuel R. Bowman and Zac Hatfield-Dodds and Ben Mann and Dario Amodei and Nicholas Joseph and Sam McCandlish and Tom Brown and Jared Kaplan
  - `year`: 2022
  - `venue`: arXiv preprint arXiv:2212.08073
  - `doi`: 10.48550/arXiv.2212.08073
  - `rename_key_to`: bai2022constitutional
- **Sources**:
  - https://arxiv.org/abs/2212.08073

### `cohen2019certified`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Venue ICML must be expanded per rules.
- **Suggested fix**:
  - `title`: Certified Adversarial Robustness via Randomized Smoothing
  - `author`: Jeremy Cohen and Elan Rosenfeld and Zico Kolter
  - `year`: 2019
  - `venue`: International Conference on Machine Learning (ICML)
  - `publisher`: PMLR
  - `pages`: 1310--1320
- **Sources**:
  - http://proceedings.mlr.press/v97/cohen19c.html

### `corbett2013spanner`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Author list is truncated. Proper noun Spanner must be protected.
- **Suggested fix**:
  - `title`: {Spanner}
  - `author`: James C. Corbett and Jeffrey Dean and Michael Epstein and Andrew Fikes and Christopher Frost and J. J. Furman and Sanjay Ghemawat and Andrey Gubarev and Christopher Heiser and Peter Hochschild and Wilson Hsieh and Sebastian Kanthak and Eugene Kogan and Hongyi Li and Alexander Lloyd and Sergey Melnik and David Mwaura and David Nagle and Sean Quinlan and Rajesh Rao and Lindsay Rolig and Yasushi Saito and Michal Szymaniak and Christopher Taylor and Ruth Wang and Dale Woodford
  - `year`: 2013
  - `venue`: ACM Transactions on Computer Systems
  - `publisher`: ACM
  - `pages`: 1--22
  - `doi`: 10.1145/2491245
- **Sources**:
  - https://dl.acm.org/doi/10.1145/2491245

### `courbariaux2016binarized`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Subtitle requires protection.
- **Suggested fix**:
  - `title`: Binarized Neural Networks: {Training} Deep Neural Networks with Weights and Activations Constrained to +1 or -1
  - `author`: Matthieu Courbariaux and Itay Hubara and Daniel Soudry and Ran El-Yaniv and Yoshua Bengio
  - `year`: 2016
  - `venue`: arXiv preprint arXiv:1602.02830
- **Sources**:
  - https://arxiv.org/abs/1602.02830

### `covington2016deep`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Proper noun YouTube must be protected.
- **Suggested fix**:
  - `title`: Deep Neural Networks for {YouTube} Recommendations
  - `author`: Paul Covington and Jay Adams and Emre Sargin
  - `year`: 2016
  - `venue`: Proceedings of the 10th ACM Conference on Recommender Systems
  - `publisher`: ACM
  - `pages`: 191--198
  - `doi`: 10.1145/2959100.2959190
- **Sources**:
  - https://dl.acm.org/doi/10.1145/2959100.2959190

### `culler1993logp`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Proper noun LogP must be protected. Subtitle requires protection.
- **Suggested fix**:
  - `title`: {LogP}: {Towards} a realistic model of parallel computation
  - `author`: David Culler and Richard Karp and David Patterson and Abhijit Sahay and Klaus Erik Schauser and Eunice Santos and Ramesh Subramonian and Thorsten von Eicken
  - `year`: 1993
  - `venue`: ACM SIGPLAN Notices
  - `publisher`: Association for Computing Machinery (ACM)
  - `pages`: 1--12
  - `doi`: 10.1145/173284.155333
- **Sources**:
  - https://dl.acm.org/doi/10.1145/173284.155333

### `dao2022flashattention`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Proper nouns FlashAttention and IO-Awareness must be protected. Subtitle requires protection. Venue NeurIPS must be expanded.
- **Suggested fix**:
  - `title`: {FlashAttention}: {Fast} and Memory-Efficient Exact Attention with {IO-Awareness}
  - `author`: Tri Dao and Daniel Y. Fu and Stefano Ermon and Atri Rudra and Christopher R\'e
  - `year`: 2022
  - `venue`: Advances in Neural Information Processing Systems (NeurIPS)
  - `publisher`: Curran Associates, Inc.
  - `pages`: 16344--16359
- **Sources**:
  - https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html

### `dastin2018amazon`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Proper nouns Amazon and AI must be protected.
- **Suggested fix**:
  - `title`: {Amazon} Scraps Secret {AI} Recruiting Tool that Showed Bias Against Women
  - `author`: Jeffrey Dastin
  - `year`: 2018
  - `venue`: Reuters
- **Sources**:
  - https://www.reuters.com/article/us-amazon-com-jobs-automation-insight/amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idUSKCN1MK08G

### `davies2011endangered`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Subtitle requires protection.
- **Suggested fix**:
  - `title`: Endangered elements: {Critical} thinking
  - `author`: Martin Davies
  - `year`: 2011
  - `venue`: Study Skills for International Postgraduates
  - `publisher`: Macmillan Education UK
  - `pages`: 111--130
  - `doi`: 10.1007/978-0-230-34553-9_8
- **Sources**:
  - https://link.springer.com/chapter/10.1007/978-0-230-34553-9_8

### `davies2018loihi`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Author list is truncated. Proper noun Loihi must be protected. Subtitle requires protection.
- **Suggested fix**:
  - `title`: {Loihi}: {A} Neuromorphic Manycore Processor with On-Chip Learning
  - `author`: Mike Davies and Narayan Srinivasa and Tsung-Han Lin and Gautham Chinya and Yongqiang Cao and Sri Harsha Choday and Georgios Dimou and Pratik Joshi and Nabil Imam and Shweta Jain and Yuyun Liao and Chit-Kwan Lin and Andrew Lines and Ruokun Liu and Dejan Markovi\'c and V. Kiran Ponulaketi and Anthony Smith and Jonathan Udell and Lida Wang
  - `year`: 2018
  - `venue`: IEEE Micro
  - `publisher`: IEEE
  - `pages`: 82--99
  - `doi`: 10.1109/mm.2018.112130359
- **Sources**:
  - https://ieeexplore.ieee.org/document/8259423

### `dayarathna2015data`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Subtitle requires protection.
- **Suggested fix**:
  - `title`: Data Center Energy Consumption Modeling: {A} Survey
  - `author`: Miyuru Dayarathna and Yonggang Wen and Rui Fan
  - `year`: 2015
  - `venue`: IEEE Communications Surveys & Tutorials
  - `publisher`: IEEE
  - `pages`: 732--794
  - `doi`: 10.1109/comst.2015.2481183
- **Sources**:
  - https://ieeexplore.ieee.org/document/7274646

### `dean2004mapreduce`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Proper noun MapReduce must be protected. Subtitle requires protection. Venue OSDI must be expanded per rules.
- **Suggested fix**:
  - `title`: {MapReduce}: {Simplified} Data Processing on Large Clusters
  - `author`: Jeffrey Dean and Sanjay Ghemawat
  - `year`: 2004
  - `venue`: USENIX Symposium on Operating Systems Design and Implementation (OSDI)
  - `publisher`: USENIX Association
  - `pages`: 1--10
- **Sources**:
  - https://www.usenix.org/legacy/events/osdi04/tech/dean.html

### `dean2012distbelief`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Author list contains an incorrect entry '0010'. Venue NIPS must be expanded per rules. Publisher should be Curran Associates.
- **Suggested fix**:
  - `title`: Large Scale Distributed Deep Networks
  - `author`: Jeffrey Dean and Greg Corrado and Rajat Monga and Kai Chen and Matthieu Devin and Quoc V. Le and Mark Z. Mao and Marc'Aurelio Ranzato and Andrew W. Senior and Paul A. Tucker and Ke Yang and Andrew Y. Ng
  - `year`: 2012
  - `venue`: Advances in Neural Information Processing Systems (NeurIPS)
  - `publisher`: Curran Associates, Inc.
  - `pages`: 1232--1240
- **Sources**:
  - https://proceedings.neurips.cc/paper/2012/hash/6aca97005c68f1206823815f66102863-Abstract.html

### `dean2012large`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Author list contains an incorrect entry '0010'. Venue NIPS must be expanded per rules. Publisher should be Curran Associates.
- **Suggested fix**:
  - `title`: Large Scale Distributed Deep Networks
  - `author`: Jeffrey Dean and Greg Corrado and Rajat Monga and Kai Chen and Matthieu Devin and Quoc V. Le and Mark Z. Mao and Marc'Aurelio Ranzato and Andrew W. Senior and Paul A. Tucker and Ke Yang and Andrew Y. Ng
  - `year`: 2012
  - `venue`: Advances in Neural Information Processing Systems (NeurIPS)
  - `publisher`: Curran Associates, Inc.
  - `pages`: 1232--1240
- **Sources**:
  - https://proceedings.neurips.cc/paper/2012/hash/6aca97005c68f1206823815f66102863-Abstract.html

### `dean2024mlsys`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Acronym AI must be protected. Subtitle requires protection.
- **Suggested fix**:
  - `title`: {AI} Hypercomputer: {Towards} an Architecture for Exascale {AI}
  - `author`: Jeff Dean
  - `year`: 2024
  - `url`: https://mlsys.org/
- **Sources**:
  - https://mlsys.org/

### `deepspeed_training_system_2021`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Key surname does not match first author (Rasley). Key year 2021 does not match paper year 2020. Proper noun DeepSpeed must be protected.
- **Suggested fix**:
  - `title`: {DeepSpeed}
  - `author`: Jeff Rasley and Samyam Rajbhandari and Olatunji Ruwase and Yuxiong He
  - `year`: 2020
  - `venue`: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining
  - `publisher`: ACM
  - `pages`: 3505--3506
  - `doi`: 10.1145/3394486.3406703
  - `rename_key_to`: rasley2020deepspeed
- **Sources**:
  - https://dl.acm.org/doi/10.1145/3394486.3406703

### `deng2009imagenet`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Proper noun ImageNet must be protected. Subtitle requires protection. Venue CVPR must be expanded per rules. Authors 'Kai Li' and 'Li Fei-Fei' should be properly formatted.
- **Suggested fix**:
  - `title`: {ImageNet}: {A} large-scale hierarchical image database
  - `author`: Jia Deng and Wei Dong and Richard Socher and Li-Jia Li and Kai Li and Li Fei-Fei
  - `year`: 2009
  - `venue`: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
  - `publisher`: IEEE
  - `pages`: 248--255
  - `doi`: 10.1109/cvpr.2009.5206848
- **Sources**:
  - https://ieeexplore.ieee.org/document/5206848

### `dennard1974design`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Acronym MOSFET's must be protected. Authors have initials instead of full first names.
- **Suggested fix**:
  - `title`: Design of ion-implanted {MOSFET's} with very small physical dimensions
  - `author`: Robert H. Dennard and Fritz H. Gaensslen and Hwa-Nien Yu and V. Leo Rideout and Ernest Bassous and Andre R. LeBlanc
  - `year`: 1974
  - `venue`: IEEE Journal of Solid-State Circuits
  - `publisher`: IEEE
  - `pages`: 256--268
  - `doi`: 10.1109/jssc.1974.1050511
- **Sources**:
  - https://ieeexplore.ieee.org/document/1050511

### `dettmers2022llm`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Acronym LLM.int8() must be protected. Subtitle requires protection.
- **Suggested fix**:
  - `title`: {LLM.int8()}: {8-bit} Matrix Multiplication for Transformers at Scale
  - `author`: Tim Dettmers and Mike Lewis and Younes Belkada and Luke Zettlemoyer
  - `year`: 2022
  - `venue`: arXiv preprint arXiv:2208.07339
- **Sources**:
  - https://arxiv.org/abs/2208.07339

### `devlin2018bert`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Acronym BERT must be protected. Subtitle requires protection.
- **Suggested fix**:
  - `title`: {BERT}: {Pre-training} of Deep Bidirectional Transformers for Language Understanding
  - `author`: Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova
  - `year`: 2018
  - `venue`: arXiv preprint arXiv:1810.04805
- **Sources**:
  - https://arxiv.org/abs/1810.04805

### `devlin2019bert`  (book/quarto/contents/vol2/backmatter/references.bib)

- **Issue**: Acronym BERT must be protected. Subtitle requires protection.
- **Suggested fix**:
  - `title`: {BERT}: {Pre-training} of Deep Bidirectional Transformers for Language Understanding
  - `author`: Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova
  - `year`: 2019
  - `venue`: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies
  - `publisher`: Association for Computational Linguistics
  - `pages`: 4171--4186
  - `doi`: 10.18653/v1/n19-1423
- **Sources**:
  - https://aclanthology.org/N19-1423/

## Uncertain (human review needed)

- `wu2019fast` (book/quarto/contents/vol1/backmatter/references.bib): Could not verify the existence of this exact paper title by Jian Wu in NeurIPS 2019. May be a hallucinated or mistitled citation.
- `cope2009pure` (book/quarto/contents/vol2/backmatter/references.bib): Missing page numbers for the journal article. Could not verify the exact pages online.

## Failed batches

- book/quarto/contents/vol1/backmatter/references.bib batch 11: failed — 50 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 6: failed — 50 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 12: failed — 50 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 5: failed — 50 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 1: failed — 50 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 10: failed — 50 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 7: failed — 50 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 0: failed — 50 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 2: failed — 50 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 8: failed — 50 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 4: failed — 50 entries skipped
- book/quarto/contents/vol2/backmatter/references.bib batch 0: failed — 50 entries skipped
- book/quarto/contents/vol2/backmatter/references.bib batch 1: failed — 50 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 3: failed — 50 entries skipped
- book/quarto/contents/vol2/backmatter/references.bib batch 3: failed — 50 entries skipped
- book/quarto/contents/vol2/backmatter/references.bib batch 4: failed — 50 entries skipped
- book/quarto/contents/vol2/backmatter/references.bib batch 5: failed — 50 entries skipped
- book/quarto/contents/vol2/backmatter/references.bib batch 6: failed — 50 entries skipped
- book/quarto/contents/vol2/backmatter/references.bib batch 7: failed — 50 entries skipped
- book/quarto/contents/vol2/backmatter/references.bib batch 8: failed — 50 entries skipped
- book/quarto/contents/vol2/backmatter/references.bib batch 9: failed — 50 entries skipped
- book/quarto/contents/vol2/backmatter/references.bib batch 10: failed — 50 entries skipped
- book/quarto/contents/vol2/backmatter/references.bib batch 11: failed — 50 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 9: failed — 50 entries skipped
- interviews/paper/references.bib batch 0: failed — 50 entries skipped
- interviews/paper/references.bib batch 1: failed — 7 entries skipped
- mlsysim/docs/references.bib batch 0: failed — 40 entries skipped
- mlsysim/paper/references.bib batch 0: failed — 50 entries skipped
- mlsysim/paper/references.bib batch 1: failed — 21 entries skipped
- periodic-table/paper/references.bib batch 0: failed — 31 entries skipped
- tinytorch/paper/references.bib batch 0: failed — 50 entries skipped
- tinytorch/paper/references.bib batch 1: failed — 3 entries skipped
- book/quarto/contents/vol1/backmatter/references.bib batch 14: failed — 21 entries skipped
