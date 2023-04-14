prepare dataset:
    
    1. pull repository from https://github.com/YasirHuang/multi30k
    2. make directory for experiments "mkdir -p experment/data"
    3. link the "multi30k" repository to "ctr_nmt/experiment/data" like: ln -s ${parent_dir}/multi30k ./ctr_nmt/experiment/data/multi30k
    
train:
    
    train with the configuration of "reconstruction source(src) language with respective(r) or shared(s) decoder, key(k) words are degraded"
    ./scripts/train/train_temmt.sh 0 src r k multi30k.parsed multi30k-test2016 
    check the results on the directory of "ctr_nmt/experiment/models/..."

requirements:

    pytorch==1.3.1 (conda install --offline pytorch-1.3.1-py3.6_cuda10.1.243_cudnn7.6.3_0.tar.bz;conda install cudatoolkit=10.1 -c pytorch / pip install torch==1.3.1)
    torchtext==0.3.1 (pip install torchtext==0.3.1)
    tqdm (pip install tqdm)
    spacy (conda install -c conda-forge spacy;python -m spacy download en_core_web_sm)
    tensorboard (pip install tensorboard)
    #nlgeval(pip install git+https://github.com/Maluuba/nlg-eval.git@master)

