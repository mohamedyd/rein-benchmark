if [ $1 = 'gpu' ];
then 
    conda install -y pytorch=1.0.1 cudatoolkit=10.0 -c pytorch
else 
    conda install -y -c pytorch pytorch=1.0.1
fi
conda install -y matplotlib=3.2.1
conda install -y pandas=0.23.4
conda install -y scikit-learn=0.22.1
conda install -y gensim=3.8.1
conda install -y tqdm=4.45.0
conda install -y -c anaconda ipykernel
python -m pip install --upgrade pip --user
pip install adversarial-robustness-toolbox==1.2.0