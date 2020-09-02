import os

os.mkdir('./Data')
os.mkdir('./weights')

Data_aug_methods = ['NO','WB','WO']

for aug in Data_aug_methods:
    os.mkdir('./Data/'+aug)
    for cnn in ['CNN1','CNN2']:
            os.mkdir('./Data/'+aug+'/'+cnn)
            os.mkdir('./Data/'+aug+'/'+cnn+'/annotest')
            os.mkdir('./Data/'+aug+'/'+cnn+'/annoval')
            os.mkdir('./Data/'+aug+'/'+cnn+'/annotrain')
            os.mkdir('./Data/'+aug+'/'+cnn+'/imagetest')
            os.mkdir('./Data/'+aug+'/'+cnn+'/imageval')
            os.mkdir('./Data/'+aug+'/'+cnn+'/imagetrain')
            os.mkdir('./Data/'+aug+'/'+cnn+'/predtest')
            os.mkdir('./Data/'+aug+'/'+cnn+'/predval')
            os.mkdir('./Data/'+aug+'/'+cnn+'/predtrain')