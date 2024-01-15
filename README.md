# ADA

This is the official implementation of our SMP2023 paper:  

Zhang Qiuling, Huayang Xu, and Jianfang Wang. [**An Adaptive Denoising Recommendation Algorithm for Causal Separation Bias**](https://link.springer.com/chapter/10.1007/978-981-99-7596-9_14) Chinese National Conference on Social Media Processing. Singapore: Springer Nature Singapore, 2023.

## Model training

First unzip the datasets and start the visdom server:
```
visdom -port 33336
```

Then simply run the following command to reproduce the experiments on corresponding dataset and model:
```
python app.py --flagfile ./config/xxx.cfg
```
# Citation

> @inproceedings{zhang2023adaptive,
  title={An Adaptive Denoising Recommendation Algorithm for Causal Separation Bias},
  author={Zhang, Qiuling and Xu, Huayang and Wang, Jianfang},
  booktitle={Chinese National Conference on Social Media Processing},
  pages={188--201},
  year={2023},
  organization={Springer}
}3}
}
