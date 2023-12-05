# Conditional Prompt Learning for Vision-Language Models (CoCoOp, CVPR'22)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2203.05557)

We provide the scripts in [scripts/cocoop](../scripts/cocoop) to reproduce CoCoOp results (CVPR'22).

Make sure to configure the dataset paths in environment variable `DATA` and run the commands from the main directory `Ensemble_VLM/`.

## Generalization From Base to New Classes

You will need both `scripts/cocoop/base2new_train.sh` and `scripts/cocoop/base2new_test.sh`. The former trains a model on bash classes while the latter evaluates the trained model on new classes.

Below we provide an example on how to evaluate the model on ImageNet with ViT-B/16. Change the config file for other vision encoders.

```bash
# seed=1
bash scripts/cocoop/base2new_train.sh imagenet 1 gpu
bash scripts/cocoop/base2new_test.sh imagenet 1 gpu new

# seed=2
bash scripts/cocoop/base2new_train.sh imagenet 2 gpu
bash scripts/cocoop/base2new_test.sh imagenet 2 gpu new

# seed=3
bash scripts/cocoop/base2new_train.sh imagenet 3 gpu
bash scripts/cocoop/base2new_test.sh imagenet 3 gpu new
```

When the evaluation is done, you can use `parse_test_res.py` to automatically calculate the average results. For instance, after you finish the evaluation (including `base2new_train.sh` and `base2new_test.sh`) on ImageNet using the aforementioned commands, you would get

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– `Config name`/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– `Config name`/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Then, to get the average performance on the base classes, run

```bash
python parse_test_res.py output/base2new/train_base/imagenet/shots_16/CoCoOp/`Config name`
```

To get the average performance on the new classes, run

```bash
python parse_test_res.py output/base2new/test_new/imagenet/shots_16/CoCoOp/`Config name` --test-log
```