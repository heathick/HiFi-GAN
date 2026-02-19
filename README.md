Проект по hifi-gan

запуск 
git clone + cd в проект
chmod +x scripts/install.sh
TORCH_FLAVOR=cpu ./scripts/install.sh

export COMET_API_KEY="твой_ключ"
python train.py -cn=baseline trainer.log_every=1
