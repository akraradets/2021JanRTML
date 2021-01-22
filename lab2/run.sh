cd AlexNetModule
echo "==== Alex ===="
python3 -u alexModule.py 
cd ..
cd GoogleNet
echo "==== Google ===="
python3 -u googlenet.py 
echo "==== GooglePretrain ===="
python3 -u googlenet_pretrain.py 
cd ..