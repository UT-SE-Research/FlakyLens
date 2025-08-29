# Commands-1: docker build -t flakylens:latest . # This command is needed to build the docker image.
# Commands-2: docker run -it --rm --gpus all flakylens:latest # This command is needed to run the docker container
## Inside the Docker container's terminal:

# To run the training code:
#bash per_project_prediction.sh FlakyLens "train"

# The model will be saved to /app/flaky-test-categorization_per_project/
# (This path resolves correctly because /app is your project root)

# To run the prediction code:
#bash per_project_prediction.sh FlakyLens "BERT"
#*********************************


# STEP 1: Choose a CUDA-enabled Base Image from NVIDIA
# Uses CUDA 12.0.1 with cuDNN 8 on Ubuntu 22.04, ideal for RTX A5000 and your specified CUDA.
FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

# Set non-interactive mode for apt-get to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set Python environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# STEP 2: Install Python 3.8.5 and pip (since you specified python=3.8.5 in your README)
# We'll use the default system Python if it's 3.8.5, or install it.
# It's often simpler to install a specific Python version using `apt-get` or `miniconda`
# For exact version control like 3.8.5, installing Miniconda is usually more straightforward.
# Let's stick with Miniconda for precise version control as implied by your README's `conda create`
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget bzip2 ca-certificates git build-essential vim && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Create a Conda environment for Python 3.8.5, as per your README
# We'll then use pip within this environment.
RUN conda create -n flakylens python=3.8.5 -y && \
    conda clean --all -f -y

# STEP 3: Set the working directory and copy files
#COPY requirements.txt .
#COPY . /app/ # Copies all other project files
WORKDIR /app
COPY . .

# Install requirements using pip from the correct environment
RUN /opt/conda/envs/flakylens/bin/pip install --upgrade pip && \
    /opt/conda/envs/flakylens/bin/pip install --no-cache-dir -r requirements.txt


# STEP 4: Install Python dependencies from requirements.txt
# This is where your requirements.txt comes in.
# If PyTorch needs a specific index-url (like for cu121), add it here.
# For example: pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
#RUN pip install --no-cache-dir -r requirements.txt

# Ensure the 'src' directory is the working directory for running scripts
WORKDIR /app/src

# Command to keep the container running so you can execute commands manually
# This is useful for interactive work, training, and running prediction scripts.
#CMD ["bash"]
#CMD ["/opt/conda/envs/flakylens/bin/bash"]
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate flakylens && bash"]

