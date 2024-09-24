# STTNet

Official Implement of the Paper "Building Extraction from Remote Sensing Images with Sparse Token Transformers"

## Some Information

[Project Page](https://kyanchen.github.io/STT/) $\cdot$ [Publication Page](https://www.mdpi.com/2072-4292/13/21/4441) $\cdot$ [HuggingFace Demo](https://huggingface.co/spaces/KyanChen/BuildingExtraction) $\cdot$ [PDF Download](http://levir.buaa.edu.cn/publications/STT_RS_.pdf)

## How to use the code

1. Prepare Data
   
   Prepare data for training, validation, and test phase. All images are with the resolution of $512 \times 512$. Please refer to the directory of **Data**.
   
   For larger images, you can patch the images with labels using **Tools/CutImgSegWithLabel.py**.
2. Get Data List
   Please refer to **Tools/GetTrainValTestCSV.py** to get the train, val, and test csv files.
3. Get Imgs Infos
   Please refer to **Tools/GetImgMeanStd.py** to get the mean value and standard deviation of the all image pixels in training set.
4. Modify Model Infos
   Please modify the model information if you want, or keep the default configuration.
5. Run to Train
   Train the model in **Main.py**.
6. [Optional] Run to Test
   Test the model with checkpoint in **Test.py**.

🚀️🚀️🚀️ We have provided pretrained models on INRIA and WHU Datasets. The pt models are in folder [Pretrain](https://huggingface.co/KyanChen/BuildingExtraction/tree/main/Pretrain).

```
@Article{rs13214441,
AUTHOR = {Chen, Keyan and Zou, Zhengxia and Shi, Zhenwei},
TITLE = {Building Extraction from Remote Sensing Images with Sparse Token Transformers},
JOURNAL = {Remote Sensing},
VOLUME = {13},
YEAR = {2021},
NUMBER = {21},
ARTICLE-NUMBER = {4441},
URL = {https://www.mdpi.com/2072-4292/13/21/4441},
ISSN = {2072-4292},
DOI = {10.3390/rs13214441}
}
```

If you have any questions, please feel free to reach me.

