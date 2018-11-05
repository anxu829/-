import zipfile , os , shutil
if os.path.exists('./VOC2018/'):
    print('exitsts ！，clean start ...')
    shutil.rmtree('./VOC2018/')
with zipfile.ZipFile('./VOC2018.zip') as zf:
    zf.extractall('.')
    
    
for  file in os.listdir('./img/'):
    shutil.copy(os.path.join('./img/',file),os.path.join('./VOC2018/JPEGImages/',file))
for  file in os.listdir('./xml//'):
    shutil.copy(os.path.join('./xml/',file),os.path.join('./VOC2018/Annotations//',file))
    
    
# 首先要做train test split
import random

imglist = set([ i.split('.')[0] for i in os.listdir('./img/')])
xmllist = set([i .split('.')[0] for i in os.listdir('./xml/')])
# 取交集
uselist = list( imglist & xmllist )
random.shuffle(uselist)

train_rate = 0.9
trainset = uselist[:int(len(uselist)*train_rate)]
testset  = uselist[int(len(uselist)*train_rate):]


with open('./VOC2018/ImageSets/Main/train.txt','w') as f  :
    for im in trainset:
        f.write(im + '\n')
        
with open('./VOC2018/ImageSets/Main/test.txt','w') as f  :
    for im in testset:
        f.write(im + '\n')
        