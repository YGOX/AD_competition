# AD_competition

The code was developed for the chinese AD competition https://www.datafountain.cn/competitions/369

MRI Data (training and testing in h5 files), Trained models, and example model outputs are available here: 
https://drive.google.com/drive/folders/1vzwF-HpwioWUvp9yPZ1qWIhx8Pjumnpo?usp=sharing

AD_3dCNN.py- baseline model a 3DCNN with VGG style, and training script embedded 

ensemble_xvali.py- integrated training and testing functions: 
1) add in cross validation 
2) add extrac metrics (e.g. F1, recall, precision) 
3) add in focal loss 
4) add in ensemble predictions: ensemble predictions from different xvalid fold to give a more robust prediction 


