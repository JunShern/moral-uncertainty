# moral-uncertainty

Dataset and baselines for the `Moral Uncertainty` benchmark. See the [Moral Uncertainty](https://junshern.github.io/moral-uncertainty/) website for details.

## Dataset
You can download the dataset files from the links in the table:
| Split | Download | # examples |
| --- | --- | --- |
| Train | [train.csv](https://drive.google.com/file/d/1je2h8QdkzC2hhBl-Mqy0lPYSKs5-Buwp/view?usp=sharing) | 13,910 |
| Test | [test.csv](https://drive.google.com/file/d/1jFXMCf0QM-QdBJnExDis8sh_BLuiEPjD/view?usp=sharing) | 2,771 |

## Baselines
The code for training our baseline models is in `moral-uncertainty/baselines`. Our specific baselines can be reproduced with:

```bash
# bert-base-uncased
python tune.py --ngpus 1 --model bert-base-uncased --learning_rate 3e-5 --batch_size 16 --nepochs 4 --gradient_acc_steps 1 --verbose
# bert-large-uncased
python tune.py --ngpus 1 --model bert-large-uncased --learning_rate 5e-6 --batch_size 16 --nepochs 4 --gradient_acc_steps 1 --verbose
# bert-base-uncased
python tune.py --ngpus 1 --model roberta-large --learning_rate 1e-5 --batch_size 16 --nepochs 4 --gradient_acc_steps 1 --verbose
# bert-base-uncased
python tune.py --ngpus 1 --model microsoft/deberta-v3-large --learning_rate 1e-5 --batch_size 16 --nepochs 4 --gradient_acc_steps 1 --verbose
# bert-base-uncased
python tune.py --ngpus 1 --model microsoft/deberta-v2-xxlarge --learning_rate 1e-6 --batch_size 8 --nepochs 4 --gradient_acc_steps 1 --verbose
```