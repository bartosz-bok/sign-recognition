# Sign Recognition using Federated Learning

This project implements Federated Learning using MLOps tools such as:
* Kubeflow
* MLFlow

As an example, it was decided to choose a Road Signs Classification task.


## Road Signs dataset

In this task **The German Traffic Sign Recognition Benchmark** (https://benchmark.ini.rub.de/gtsrb_news.html) was chosen.



## Road Signs Recognition script

https://debuggercafe.com/traffic-sign-recognition-using-pytorch-and-deep-learning/
```bash
source config.sh
python3 sign_recognition_script/src/train.py --pretrained --fine-tune --epochs $EPOCHS --learning-rate $LEARNING_RATE
```

This repository contains `src/` catalog. But if you want to run this script, be sure, that you have directory structure as presented below:

```yaml
├── input
│   ├── GTSRB_Final_Test_GT
│   │   └── GT-final_test.csv
│   ├── GTSRB_Final_Test_Images
│   │   └── GTSRB
│   │       ├── Final_Test
│   │       │   └── Images [12631 entries exceeds filelimit, not opening dir]
│   │       └── Readme-Images-Final-test.txt
│   ├── GTSRB_Final_Training_Images
│   │   └── GTSRB
│   │       ├── Final_Training
│   │       │   └── Images
│   │       │       ├── 00000 [211 entries exceeds filelimit, not opening dir]
│   │       │       ├── 00001 [2221 entries exceeds filelimit, not opening dir]
                    ...
│   │       │       ├── 00040 [361 entries exceeds filelimit, not opening dir]
│   │       │       ├── 00041 [241 entries exceeds filelimit, not opening dir]
│   │       │       └── 00042 [241 entries exceeds filelimit, not opening dir]
│   │       └── Readme-Images.txt
│   ├── README.txt
│   └── signnames.csv
├── outputs
│   ├── test_results [12630 entries exceeds filelimit, not opening dir]
│   ├── accuracy.png
│   ├── loss.png
│   └── model.pth
└── src
    ├── cam.py
    ├── datasets.py
    ├── model.py
    ├── train.py
    └── utils.py
```

*QUESTIONS*:
* why we use ready model `mobilenet_v3`
* what is the difference of `mobilenet_v3_large` and `mobilenet_v3_small`
* what means `pretrained` flag
* what means `fine_tune` flag
* what means `scheduler` flag

## Federated Learning scenarios

## Federated Learning implementation

## Kubeflow

### Defining component as docker image

https://medium.com/ubuntu-ai/how-to-build-and-share-components-for-kubeflow-pipelines-86f2c8f40de5
