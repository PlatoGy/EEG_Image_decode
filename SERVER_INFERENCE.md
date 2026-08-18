# Server Inference Run Order

Assumed server paths:

```bash
repo=/data/gaoy/projects/EEG_Image_decode
data=/data/gaoy/projects/datasets/EEG_Image_decode
```

Start from a fresh shell after `git pull`:

```bash
conda activate BCI
cd /data/gaoy/projects/EEG_Image_decode
export EEG_IMAGE_DECODE_REPO=/data/gaoy/projects/EEG_Image_decode
export EEG_IMAGE_DECODE_DATA=/data/gaoy/projects/datasets/EEG_Image_decode
export HF_ENDPOINT=https://hf-mirror.com
```

1. Launch Jupyter from the Generation directory:

```bash
cd /data/gaoy/projects/EEG_Image_decode/Generation
jupyter notebook --ip=0.0.0.0 --no-browser
```

2. Open `Generation_metrics_sub8.ipynb`.

3. Use released Subject-08 EEG embeddings when you want to run inference before training ATMS:

Keep this notebook setting:

```python
USE_RELEASED_EEG_EMBEDDINGS = True
```

This loads:

```text
$EEG_IMAGE_DECODE_DATA/emb_eeg/ATM_S_eeg_features_sub-08_test.pt
$EEG_IMAGE_DECODE_DATA/emb_eeg/ATM_S_eeg_features_sub-08_train.pt
```

The raw EEG to ATMS encoder step needs the checkpoint produced by ATMS training:

```text
models/contrast/ATMS/02-01_00-39/sub-08/40.pth
```

Do not substitute `fintune_ckpts/sub-08/diffusion_prior.pt` for this checkpoint.

4. Run Stage-I diffusion prior:

The notebook loads the released diffusion prior from:

```text
$EEG_IMAGE_DECODE_DATA/fintune_ckpts/sub-08/diffusion_prior.pt
```

It is loaded only into `DiffusionPriorUNet` from `Generation/diffusion_prior.py`.

5. Run SDXL/IP-Adapter reconstruction:

Continue the generation cells in `Generation_metrics_sub8.ipynb`, or open `1x1024_reconstruct_sdxl.ipynb` for the low-level SDXL/IP-Adapter reconstruction path. Hugging Face model IDs are unchanged; `HF_ENDPOINT` is supplied by the shell environment.

6. Alternative no-`40.pth` script:

This follows the same released-embedding inference path without instantiating ATMS:

```bash
cd /data/gaoy/projects/EEG_Image_decode
python Generation/generate_sub8_from_released_embeddings.py --device cuda:0
```

Useful partial run while checking the server:

```bash
python Generation/generate_sub8_from_released_embeddings.py --device cuda:0 --start 0 --count 2 --repeats 1
```

7. Run metrics:

Open `Reconstruction_Metrics_ATM.ipynb`. It reads generated images from:

```text
$EEG_IMAGE_DECODE_DATA/generated_imgs/sub-08
```

and ground-truth test images from:

```text
$EEG_IMAGE_DECODE_DATA/images_set/test_images
```
