python setup.py develop

python inference.py --device cpu
python inference.py --device cuda --batch_size 1 --channels_last 1 --precision float16 --jit --nv_fuser
