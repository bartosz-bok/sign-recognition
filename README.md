# Sign Recognition using Federated Learning

As an example, it was decided to choose a Road Signs Classification task.


## Road Signs dataset

In this task **The German Traffic Sign Recognition Benchmark** (https://benchmark.ini.rub.de/gtsrb_news.html) was chosen.



## Road Signs Recognition script

Training scripts are based on internet article:
https://debuggercafe.com/traffic-sign-recognition-using-pytorch-and-deep-learning/

This repository contains `src/` catalog, which includes training scripts based on article.
If you want to run training script, be sure, that you have directory structure as presented below:

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

Run script to train models:
```bash
source config.sh
python3 sign_recognition_script/src/train.py --pretrained --fine-tune --epochs $EPOCHS --learning-rate $LEARNING_RATE
```

*QUESTIONS* (answers in polish to better understanding):
* why we use ready model `mobilenet_v3` - Mobilenet v3 jest znany z tego, że jest lekki i wydajny, co oznacza, że może
być dobrze dostosowany do środowisk o ograniczonych zasobach obliczeniowych, takich jak aplikacje mobilne lub urządzenia
wbudowane. Jest to istotne w przypadku systemów wizyjnych, takich jak rozpoznawanie znaków drogowych, które muszą
działać w czasie rzeczywistym na urządzeniach o ograniczonych zasobach.
Mobilenet v3 różni się od poprzednich wersji m.in. dzięki zastosowaniu technik optymalizacyjnych, które pozwalają na
jeszcze lepsze wykorzystanie zasobów sprzętowych. Dlatego Mobilenet v3 może być atrakcyjną opcją dla zadań, w których
liczy się wydajność i efektywność, bez konieczności rezygnowania z jakości rozpoznawania.
* what is the difference of `mobilenet_v3_large` and `mobilenet_v3_small` - różnica polega na rozmiarze i złożoności
modelu. mobilenet_v3_small ma mniej parametrów i jest mniejszy od mobilenet_v3_large, co oznacza, że jest bardziej
lekki i może być szybciej wdrażany na urządzeniach z mniejszą mocą obliczeniową. Z drugiej strony mobilenet_v3_large
ma więcej parametrów i jest bardziej złożony, co może prowadzić do lepszej dokładności, zwłaszcza w bardziej
zróżnicowanych zbiorach danych, ale może być też bardziej wymagający obliczeniowo.
* parameters in `mobilenet_v3`:
* * `pretrained` - wskazuje, czy model powinien zostać zainicjalizowany przy użyciu wag wstępnie wytrenowanych.
Wagi wstępnie wytrenowane są to parametry modelu, które zostały już wytrenowane na dużej i ogólnej bazie danych,
zazwyczaj umożliwiając modelowi lepsze radzenie sobie z zadaniami specyficznymi bez konieczności trenowania od zera.
Użycie wstępnie wytrenowanych wag jest szczególnie pomocne w przypadku, gdy dysponujemy ograniczonymi zasobami danych
lub obliczeniowymi.
* * `fine_tune` - pozwala na dokładniejsze dostosowanie modelu do konkretnego zadania. Jeśli ta opcja jest
aktywna, model będzie trenowany (dostrajany) na specyficznych dla zadania danych, co pozwala na dalszą optymalizację
parametrów modelu, które zostały wstępnie wytrenowane. Dostrajanie może obejmować cały model lub tylko jego część, w
zależności od potrzeb i dostępnych danych. To pozwala na poprawę skuteczności modelu na bardziej specyficznych danych
lub zadaniach.
* * `scheduler` -  jest mechanizmem kontrolującym szybkość uczenia się modelu (learning rate) w czasie. W przykładzie
użyto CosineAnnealingWarmRestarts, który jest rodzajem harmonogramu szybkości uczenia. Harmonogram ten zmniejsza
szybkość uczenia się według funkcji cosinusowej między pewnymi wartościami maksymalnymi i minimalnymi w określonych
cyklach, a następnie "restartuje" ten proces, co może pomóc w uniknięciu lokalnych minimów i poprawie ogólnej
skuteczności trenowania modelu. Parametry T_0 i T_mult określają długość każdego cyklu i sposób, w jaki cykle się
zmieniają.

## Federated Learning scenarios

## Federated Learning implementation

