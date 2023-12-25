# C-VTON

[Original Repository](https://github.com/benquick123/C-VTON)

# Test Dataset

https://drive.google.com/file/d/1_8LhD5OjdVbJCmBcjpPqiSVxKCOO2wZU/view?usp=sharing

This zip file contains both the input data and and the outputs generated from the model during our test run.

Input data is in **viton/data** folder.  
Generated outputs are in **viton/results** folder.

# Recommended Environment

This was the environment in which we ran the experiments:  

**Linux-x64**  
**Python v3.11.5**  
**Torch v2.1.2**  
**Torchvision v0.16.2**  
**Without CUDA, CPU only**

# How to run

If you want to generate the outputs on your own machine, follow these steps:

1. Clone the repository and move into the cloned directory.
```bash
git clone https://github.com/VTON-Project/C-VTON.git
cd C-VTON
```

2. Download [test_data.zip](https://drive.google.com/file/d/1_8LhD5OjdVbJCmBcjpPqiSVxKCOO2wZU/view?usp=sharing) file. Create a **'data'** folder inside the C-VTON folder. Extract the zip file into this folder.

3. Go to the [original repository](https://github.com/benquick123/C-VTON#testing) and download the BPGM and C-VTON pretrained models for VITON-HD as instructed in the **Testing** section. Put the models in their respective folders.

4. [Install PyTorch](https://pytorch.org/get-started/locally/).

5. Install other required packages in your environment using requirements.txt file.
```bash
pip install -r requirements.txt
```

6. Run **scripts/test_vitonhd.sh**.
```bash
source scripts/test_vitonhd.sh
```

7. You will find the generated outputs in **results/C-VTON-VITON-HD/test_images** folder.