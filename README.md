# Multi-Game PCGRL
Multi-game Procedural Content Generation via Reinforcement Learning project


## Environment Setup

### Build Docker Image

```bash
docker build -t bic4907/multigame .
```



## How to Run

### Using Docker

The `run_docker.sh` script automatically finds and allocates an available GPU, and creates container names in the format `multigame-pcgrl_gpu{GPU_NUMBER}_{DATE}`.

#### Basic Usage

```bash
./run_docker.sh <command> [args...]
```


#### Weights & Biases Setup
Create a `.env` file in the root directory with the following content:
```
WANDB_API_KEY=your_wandb_api_key
```

### Examples

**1. Integrated Training (CLIP + VQ-VAE)**
```bash
./run_docker.sh python train.py exp_name=my_experiment n_epochs=300 batch_size=128
```

**2. CLIP Only Training**
```bash
./run_docker.sh python train_clip.py exp_name=clip_only n_epochs=100 batch_size=128 lr=5e-5
```

**3. VQ-VAE Only Training**
```bash
./run_docker.sh python train_cvae.py exp_name=cvae_only n_epochs=100 batch_size=512 lr=2e-4
```

**4. Specify GPU**
```bash
GPU=2 ./run_docker.sh python train.py exp_name=my_experiment
```

**5. Select Games**
```bash
./run_docker.sh python train.py games=smb_tloz_lr_dg
```

