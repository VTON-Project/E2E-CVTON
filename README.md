# C-VTON

[Original Repository](https://github.com/benquick123/C-VTON)

## Assets

### Test Dataset

[Download](https://drive.google.com/file/d/1_8LhD5OjdVbJCmBcjpPqiSVxKCOO2wZU/view?usp=sharing)

This zip file contains both the input data and and the outputs generated from the model during our test run.

Input data is in **viton/data** folder.  
Generated outputs are in **viton/results** folder.

### Masking Model

[Download](https://drive.google.com/file/d/1cFtdCElcNKmS62vGJ65I9mVJH9m2m75M/view?usp=sharing)

Trained parameters for `masking_model.Masker`.

## Recommended Environment

This was the development environment which we used:

- **Linux-x64**  
- **Python v3.12.2**  
- **Torch v2.2.1**  
- **Torchvision v0.17.1**  

## How to run

1. Clone the repository and move into the cloned directory.
    ```bash
    git clone https://github.com/VTON-Project/C-VTON.git
    cd C-VTON
    ```

2. Go to the [original repository](https://github.com/benquick123/C-VTON#testing) and download the BPGM and C-VTON pretrained models for VITON-HD as instructed in the **Testing** section. Put the models in their respective folders.

3. Download [masking_model.zip](https://drive.google.com/file/d/1cFtdCElcNKmS62vGJ65I9mVJH9m2m75M/view?usp=sharing) file and extract its contents into the ***masking_model*** directory.

4. [Install PyTorch](https://pytorch.org/get-started/locally/).

5. Install the required packages in your environment using requirements.txt file.
    ```bash
    pip install -r requirements.txt
    ```

6. Run the following command to start a gunicorn server:

    ```bash
    gunicorn api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 127.0.0.1:5000
    ```

6. Now you will be able to access the API at *http://localhost:5000*. To initiate the process, send a POST request to the API containing two images with keys 'person' and 'cloth' respectively. Upon receipt, the server will generate a try-on image in response. *You can use a tool like Postman for this.*
