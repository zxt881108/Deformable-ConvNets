import cPickle
CLASSES = ['__background__',  # always index 0
           'tibetan flag', 'guns','knives','not terror','islamic flag','isis flag']

cache_file = 'qiniuV5_test_detections.pkl'

thresh_old = 0.9
 

with open('qiniuv5.txt') as fid:
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
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(y_true, y_scores)
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_true, y_scores)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
          average_precision))
plt.savefig('1.jpg')
