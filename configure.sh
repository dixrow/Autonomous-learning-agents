cd workspace/

git clone https://github.com/dixrow/aal.git
cd tfm_agentes_autonomos

python -m venv .venv
source .venv/bin/activate

python -m install ipykernel --user --name aal

pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

apt update
apt install gh
apt install vim
