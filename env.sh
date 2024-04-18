export TRANSFORMERS_CACHE=/scratch/$USER/.cache
export HF_HOME=/scratch/$USER/.cache
echo "make sure to export HF_TOKEN=\"your_api_token\""
echo TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE
echo HF_HOME=$HF_HOME

module load gcc/11.4.0 openmpi/4.1.4 python/3.11.4
python -m venv ENV
source ENV/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
