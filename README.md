## The Pitfalls of Simplicity Bias in Neural Networks

### Summary

This repository consists of code primitives and Jupyter notebooks that can be used to replicate and extend the findings presented in the paper "The Pitfalls of Simplicity Bias in Neural Networks" ([link](https://arxiv.org/abs/2006.07710)). In addition to the code (in scripts/) to generate the proposed datasets, we provide five Jupyter notebooks:

1. ```01_extremeSB_slab_data.ipynb``` shows the simplicity bias of fully-connected networks trained on synthetic slab-structured datasets.
2. ```02_extremeSB_mnistcifar_data.ipynb``` highlights simplicity bias of commonly-used convolutional neural networks (CNNs) on the concatenated MNIST-CIFAR dataset,
3. ```03_suboptimal_generalization.ipynb``` analyzes the effect of extreme simplicity bias on standard generalization.
4. ```04_effect_of_ensembles.ipynb``` studies the effectiveness of ensembles of independently trained methods in mitigating simplicity bias and its pitfalls.
5. ```05_effect_of_adversarial_training.ipynb``` evaluates the effectiveness of adversarial training in mitigating simplicity bias. 

Please check out our [paper](https://arxiv.org/abs/2006.07710) or [poster](https://drive.google.com/file/d/10McXcIyTM8pxJE2edqcvO2cBxmq8is2P/view?usp=sharing) for more details.  

###  Setup

Our code uses Python 3.7.3, Torch 1.1.0, Torchvision 0.3.0, Ubuntu 18.04.2 LTS and the packages listed in `requirements.txt`.

---

If you find this project useful in your research, please consider citing the following publication:

```
@article{shah2020pitfalls,
  title={The Pitfalls of Simplicity Bias in Neural Networks},
  author={Shah, Harshay and Tamuly, Kaustav and Raghunathan, Aditi and Jain, Prateek and Netrapalli, Praneeth},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

