# AD_competition

The code was developed for the chinese AD competition https://www.datafountain.cn/competitions/369

MRI Data (training and testing in h5 files), Trained models, and example model outputs are available here: 
https://drive.google.com/drive/folders/1vzwF-HpwioWUvp9yPZ1qWIhx8Pjumnpo?usp=sharing

AD_3dCNN.py- baseline model a 3DCNN with VGG style, and training script embedded 

test.py- baseline model testing and the predicted labels on the test data are kept in a .csv file 

ensemble_xvali.py- integrated training and testing functions: 
1) add in cross validation 
2) add extrac metrics (e.g. F1, recall, precision) 
3) add in focal loss 
4) add in ensemble predictions: ensemble predictions from different xvalid fold to give a more robust prediction 

ensemble_test.py- testing script only, ensemble the predictions from available xvalidated models

callback.py- customerised F1 score based callback function for keras.fit() 

vis_mri.py- visualize slice-wise MRI slices 

vis_saliency.py- guided-backpropogation to attain a pixel-wise activation map (saliency) and visualize slice-wise saliency hilighting the regions related with the predicted disease label






