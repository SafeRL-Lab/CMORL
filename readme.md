 ## Installation of the Environments

1. **Create an environment (requires [Conda installation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)):**
We are currently developing our environments using a Linux system. 

   Use the following command to create a new Conda environment named `cmorl` with Python 3.11:

   ```bash
   conda create -n cmorl  python=3.9
   ```

   Activate the newly created environment:

   ```bash
   conda activate cmorl
   ```

3. **Install dependency packages:**

   Install the necessary packages using pip. Make sure you are in the project directory where the `setup.py` file is located:

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
---------------
## Run experiments

To run the experiments, run test script, e.g.,

```bash
`./train_mujoco`
```

---------


## Citation
If you find the repository useful, please cite the study
``` Bash
@article{gu2025safe,
  title={Safe and balanced: A framework for constrained multi-objective reinforcement learning},
  author={Gu, Shangding and Sel, Bilgehan and Ding, Yuhao and Wang, Lu and Lin, Qingwei and Knoll, Alois and Jin, Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```

-------



We thank the contributors from GitHub open source repositories [1](https://github.com/ikostrikov/pytorch-trpo), [2](https://github.com/Cranial-XIX/CAGrad), [3](https://github.com/google-deepmind/mujoco), and [4](https://github.com/openai/mujoco-py).
