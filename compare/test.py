import cPickle
CLASSES = ['__background__',  # always index 0
           'tibetan flag', 'guns','knives','not terror','islamic flag','isis flag']

cache_file = 'PascalVOC_2007_test_weibo_detections.pkl'

thresh_old = 0.9
 

with open('test.txt') as fid:
    filenamelist = fid.readlines()
fout = open('res.txt','w')

with open(cache_file, 'rb') as fid:
    all_boxes  = cPickle.load(fid)
print(len(all_boxes[0]))
print(len(filenamelist))
for i in range(0,len(all_boxes[0])):
#    for h in xrange
    for j in xrange(1, len(CLASSES)):
        if CLASSES[j] == 'not terror':
            continue
        boxes  = all_boxes[j][i] 
        for box in boxes:
            det_score = box[4]
            if det_score > thresh_old:
                line = filenamelist[i].strip() + ' ' + CLASSES[j] + ' ' + str(det_score) + ' ('+str(box[0]) + ',' + str(box[1]) + ','+str(box[2])+','+ str(box[3])+')\n'
                fout.write(line)
#    print(boxes_this_image)
fout.close()

