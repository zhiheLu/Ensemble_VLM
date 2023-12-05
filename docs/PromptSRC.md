# PromptSRC Training

The code has been borrowed from [PromptSRC](https://github.com/muzairkhattak/PromptSRC).

We provide bash scripts in [scripts/](../scripts) for training PromptSRC.

Make sure to update the `DATA` variable with dataset path in the script file and run the commands from the main directory `Ensemble_VLM/`.
Below we provide training and testing instructions for PromptSRC.

## PromptSRC

#### (1) Base-to-Novel class generalization setting
The base-to-novel PromptSRC configurations are provided in config file at `configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml` for ViT-B/16 and `configs/trainers/PromptSRC/vit_b32_c2_ep20_batch4_4+4ctx.yaml` for ViT-B/32.

Run the commands below to train PromptSRC on ImageNet with ViT-B/16. Change the config for other vision encoders.

```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# seed=1
# trains and evaluates on base classes
bash scripts/promptsrc/base2new_train.sh imagenet 1 gpu
# evaluates on novel classes
bash scripts/promptsrc/base2new_test.sh imagenet 1 gpu new

# seed=2
# trains and evaluates on base classes
bash scripts/promptsrc/base2new_train.sh imagenet 2 gpu
# evaluates on novel classes
bash scripts/promptsrc/base2new_test.sh imagenet 2 gpu new

# seed=3
# trains and evaluates on base classes
bash scripts/promptsrc/base2new_train.sh imagenet 3 gpu
# evaluates on novel classes
bash scripts/promptsrc/base2new_test.sh imagenet 3 gpu new
```

#### Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– PromptSRC/
|   |   |   |   |   |–– `Config name`/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– PromptSRC/
|   |   |   |   |   |–– `Config name`/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Now use the script `parse_test_res.py` and run the commands below to calculate the averaged results:
```bash
# prints averaged results for base classes
python output/base2new/train_base/imagenet/shots_16/PromptSRC/`Config name` --test-log
# averaged results for novel classes
python output/base2new/test_new/imagenet/shots_16/PromptSRC/`Config name` --test-log
```

The above steps can be repeated for other individual datasets.