# Neural Invertible Warp for NeRF (ECCV-2024)

<a href='https://sfchng.github.io/ineurowarping-github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2407.12354'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

This is an official implementation of the paper "Neural Invertible Warp for NeRF".


## 🛠️ Installation Steps
Assuming a fresh Anaconda environment, you can install the dependencies by
```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
numpy==1.26.4

pip install -r requirements.txt

conda install -c conda-forge cupy cuda-version=11.6
```

## 💿 Training data

<details>
  <summary><b>DTU</b></summary>

* Images: We use the DTU dataset, produced by SPARF, where the images are processed and resized to 300 x 400.
Download the data [here](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR?usp=sharing). 

* Ground-truth depth maps: For geometry evaluation, we report the depth error. Download the [depth maps](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip). They are extracted from [MVSNeRF](https://github.com/apchenstu/mvsnerf#:~:text=training%20data%20and-,Depth_raw,-from%20original%20MVSNet).  

</details>

<details>
  <summary><b>LLFF</b></summary>

The LLFF real-world data can be found in the [NeRF Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
You can download the dataset by running
```shell
gdown 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g # download nerf_llff_data.zip
unzip nerf_llff_data.zip
rm -f nerf_llff_data.zip
mv nerf_llff_data data/llff
```

</details>


The **data** directory should contain the subdirectories **llff** and **dtu**. If you have downloaded the datasets, you can create soft links to them within the **data** directory.

## ⏳ Training and Evaluation

By default, models and TensorBoard event files are saved to `~/output/<GROUP>/<NAME>`. This can be modified using the `--output_root` flag.

| Dataset   | Training script  | Eval script     | Notes                                   |
| --------- | ---------------- | --------------- | -------------------------               |
| LLFF      | `train_llff.sh`  | `eval_llff.sh`  | Matches Table 1 in supplementary paper  |
| DTU       | `train_dtu.sh`   | `eval_dtu.sh`   | Matches Table 2 in supplementary paper  |

## 👩‍💻 Citation
This code is for non-commercial use.
If you find our work useful in your research please cite our paper:
```
@inproceedings{chng2024invertible,
  title={Invertible neural warp for nerf},
  author={Chng, Shin-Fang and Garg, Ravi and Saratchandran, Hemanth and Lucey, Simon},
  booktitle={European Conference on Computer Vision},
  pages={405--421},
  year={2024},
  organization={Springer}
}
```
