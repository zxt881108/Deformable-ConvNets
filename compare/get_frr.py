import cPickle
CLASSES = ['__background__',  # always index 0
           'tibetan flag', 'guns','knives','not terror','islamic flag','isis flag']

cache_file = 'v5_weibo_neg.pkl'

thresh_old = 0.95


with open('test.txt') as fid:
    filenamelist = fid.readlines()
fout = open('res.txt','w')

with open(cache_file, 'rb') as fid:
    all_boxes  = cPickle.load(fid)
print(len(all_boxes[0]))
print(len(filenamelist))

y_true =[]
y_scores = []
for i in range(0,len(all_boxes[0])):
#    for h in xrange
    if filenamelist[i].strip().split("/")[-1][0] == '1':
        y_true.append(1)
    else:
        y_true.append(0)      
    max_score =0
    for j in xrange(1, len(CLASSES)):
        if CLASSES[j] == 'not terror':
            continue
        boxes  = all_boxes[j][i] 
        for box in boxes:
            det_score = box[4]
            if det_score > max_score:
                max_score = det_score
            if det_score > thresh_old:
                line = filenamelist[i].strip() + ' ' + CLASSES[j] + ' ' + str(det_score) + ' ('+str(box[0]) + ',' + str(box[1]) + ','+str(box[2])+','+ str(box[3])+')\n'
                fout.write(line)
#    print(boxes_this_image)
    y_scores.append(max_score)
fout.close()
import numpy as np
y_scores = np.array(y_scores)
thresh_olds = [0.7,0.8,0.9,0.95]
for thresh_old in thresh_olds:
    print('FRR @' + str(thresh_old) + ' is ' + str(len(y_scores[y_scores>thresh_old])/float(len(filenamelist))))
