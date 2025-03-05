# SD2AIL: Adversarial Imitation Learning from Synthetic Demonstrations via Diffusion Models

Code for "SD2AIL: Adversarial Imitation Learning from Synthetic Demonstrations via Diffusion Models"



# Environments

SD2AIL is evaluated  on MuJoCo continuous control tasks in OpenAI gym. It is trained using PyTorch 1.11.0+cu113 and Python
3.8.

Install Dependencies
```
pip install -r requirements.txt
```
You will also need to install Mujoco and use a valid license. Follow the install instructions [here](https://github.com/openai/mujoco-py).

# Dataset
All dataset files are sourced from [DiffAIL](https://github.com/ML-Group-SDU/DiffAIL)

# Usage


The paper results can be reproduced by running:
```
python ./run_experiment.py -e "./exp_specs/ddpm_halfcheetah.yaml" -g 0  # for SD2AIL
```

The algorithms we used BC, GAIL, CFIL and DiffAIL can be found from the following sources:

BC[(code)](https://github.com/google-research/google-research/tree/master/value_dice): Behavior cloning   
CFIL[(code)](https://github.com/gfreund123/cfil) : A Coupled Flow Approach to Imitation Learning[(paper)](https://arxiv.org/abs/2305.00303)  
GAIL + DiffAIL[(code)](https://github.com/ML-Group-SDU/DiffAIL): DiffAIL: Diffusion Adversarial Imitation Learning[(paper)](https://arxiv.org/abs/2312.06348)


# Acknowledgements
This repo relies on the following existing codebases:
- The overall framework is based on [DiffAIL](https://github.com/ML-Group-SDU/DiffAIL)
- The diffusion model variant  based on [Diffusion Q](https://github.com/zhendong-wang/diffusion-policies-for-offline-rl)
- The Adversarial Imitation Learning framework is adapted from [here](https://github.com/Ericonaldo/ILSwiss)

# Citation
If you use this code for your research, please consider citing the paper:
```

```
