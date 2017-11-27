import _init_paths
import argparse
import os
import cv2
import cPickle
import sys
import mxnet as mx
import numpy as np
from config.config import config, update_config
from symbols import *
from core.tester import Predictor, im_detect, im_proposal, vis_all_detection, draw_all_detection
from utils.load_model import load_param
from utils.create_logger import create_logger
from utils.image import resize, transform
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from nms.box_voting import py_box_voting_wrapper
import json
from multiprocessing import Pool

pool = Pool(processes=25)


CLASSES = ['__background__',  # always index 0
           'tibetan flag', 'guns','knives','not terror','islamic flag','isis flag']


def generate_batch(im):
    """
    preprocess image, return batch
    :param im: cv2.imread returns [height, width, channel] in BGR
    :return:
    data_batch: MXNet input batch
    data_names: names in data_batch
    im_scale: float number
    """
    SHORT_SIDE = config.SCALES[0][0]
    LONG_SIDE = config.SCALES[0][1]
    PIXEL_MEANS = config.network.PIXEL_MEANS
    DATA_NAMES = ['data', 'im_info']
    im_array, im_scale = resize(im, SHORT_SIDE, LONG_SIDE)
    im_array = transform(im_array, PIXEL_MEANS)
    im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
    data = [[mx.nd.array(im_array), mx.nd.array(im_info)]]
    data_shapes = [[('data', im_array.shape), ('im_info', im_info.shape)]]
    data_batch = mx.io.DataBatch(data=data, label=[None], provide_data=data_shapes, provide_label=[None])
    return data_batch, DATA_NAMES, [im_scale]

def generate_batch_V2(im,num_gpu):
    """
    preprocess image, return batch
    :param im: cv2.imread returns [height, width, channel] in BGR
    :return:
    data_batch: MXNet input batch
    data_names: names in data_batch
    im_scale: float number
    """
    array_list , info_list =[],[]
    for idx in range(0,num_gpu):
        SHORT_SIDE = config.SCALES[0][0]
        LONG_SIDE = config.SCALES[0][1]
        PIXEL_MEANS = config.network.PIXEL_MEANS
        DATA_NAMES = ['data', 'im_info']
        im_array, im_scale = resize(im[idx], SHORT_SIDE, LONG_SIDE)
        im_array = transform(im_array, PIXEL_MEANS)
        im_info = np.array([[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
        array_list.append(im_array)
        info_list.append(im_info)

    data = [[mx.nd.array(_), mx.nd.array(__)] for _ in array_list,for __ in info_list]
    data_shapes = [[('data', array.shape), ('im_info', __.shape)] for array in array_list,for __ in info_list]
    data_batch = mx.io.DataBatch(data=data, label=[None], provide_data=data_shapes, provide_label=[None])
    return data_batch, DATA_NAMES, [im_scale]


def image_path_from_index(index, dataset_path, image_set):
    """
    given image index, find out full path
    :param index: index of a specific image
    :return: full path of this image
    """
    image_file = os.path.join(dataset_path, 'DET', 'Data','DET', image_set, index + '.JPEG')
    assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
    return image_file

def save_vis(image_file,data_dict,boxes_this_image,im_scale,cfg):
    vis_image_dir = 'vis'
    if not os.path.exists(vis_image_dir):
        os.mkdir(vis_image_dir)
    result_file = os.path.join(vis_image_dir, image_file.strip().split('/')[-1] + '_result' + '.JPEG')
    print('results saved to %s' % result_file)
    im = draw_all_detection(data_dict[0]['data'].asnumpy(), boxes_this_image, CLASSES, im_scale, cfg)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imwrite(result_file, im)



def predict_url(cfg,predictor,image_set_index,nms,box_voting,all_boxes,jsonlistwithscore ,jsonlistwithoutscore,
                       thresh, vis=False, use_box_voting=False,num_gpu=1):
    """
	  generate data_batch -> im_detect -> post process
	  :param predictor: Predictor
	  :param image_name: image name
	  :param vis: will save as a new image if not visualized
	  :return: None
	  """

    #   image_file = image_path_from_index(index, dataset_path, image_set)

    for idx in range(0,len(image_set_index)/num_gpu):
        import urllib
        im = []
        for j in range(0,num_gpu):
            image_file = image_set_index[num_gpu * idx + j]
            proxies = {'http': 'http://xsio.qiniu.io'}
            data = urllib.urlopen(image_file.strip(), proxies=proxies).read()
            nparr = np.fromstring(data, np.uint8)
            im.append(cv2.imdecode(nparr, 1))
            #        im = cv2.imread(image_file)/

        box_list =[]
        data_batch, data_names, im_scale = generate_batch_V2(im,num_gpu)
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, im_scale, config)
        for cls in CLASSES:
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[0][:, 4:8] if config.CLASS_AGNOSTIC else boxes[0][:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[0][:, cls_ind, np.newaxis]
            keep = np.where(cls_scores >= thresh)[0]
            cls_dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]

            keep = nms(cls_dets)

            # apply box voting after nms
            if use_box_voting:
                nms_cls_dets = cls_dets[keep, :]
                all_boxes[cls_ind][idx] = box_voting(nms_cls_dets, cls_dets)
            else:
                all_boxes[cls_ind][idx] = cls_dets[keep, :]

        boxes_this_image = [[]] + [box_list[j] for j in xrange(1, len(CLASSES))]

        dets, dets_s = [], []
        for j in xrange(1, len(CLASSES)):
            if CLASSES[j] == 'not terror':
                continue
            boxes = box_list[j]
            for box in boxes:
                det_score = box[4]
                if det_score > thresh:
                    det, det_s = dict(), dict()
                    xmin = float(box[0])
                    ymin = float(box[1])
                    xmax = float(box[2])
                    ymax = float(box[3])
                    det['pts'] = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                    det['class'] = CLASSES[j]
                    det_s['pts'] = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                    det_s['class'] = CLASSES[j]
                    det_s['score'] = float(det_score)
                    # det['score'] = float(det_score)
                    # det['bbox'] = [float(box[0]),float(box[1]),float(box[2]),float(box[3])]
                    # det['class'] = CLASSES[j]

                    dets.append(det)
                    dets_s.append(det_s)
        # line = {}
        # line['detections'] = dets
        # line['img'] = image_file

        ress = {
            "url": image_file,
            "label": {"detect": {"general_d": {"bbox": dets}}},
            "type": "image",
            "source_url": "",
            "ops": "download()"
        }

        ress_s = {
            "url": image_file,
            "label": {"detect": {"general_d": {"bbox": dets_s}}},
            "type": "image",
            "source_url": "",
            "ops": "download()"
        }

        if vis:
            # vis_all_detection(data_dict['data'].oasnumpy(), boxes_this_image, CLASSES, im_scale)
            save_vis(image_file, data_dict, boxes_this_image, im_scale, cfg)


        jsonlistwithscore[idx] = json.dumps(ress_s)
        jsonlistwithoutscore[idx] = json.dumps(ress)

    return


def demo_net(cfg,predictor, dataset, image_set,
             root_path, dataset_path, thresh, vis=False, use_box_voting=False,
             test_file='test.txt',out_prefix='output',vis_image_dir='vis'):
    """
    generate data_batch -> im_detect -> post process
    :param predictor: Predictor
    :param image_name: image name
    :param vis: will save as a new image if not visualized
    :return: None
    """
    # visualization
    nms = py_nms_wrapper(config.TEST.NMS)
    box_voting = py_box_voting_wrapper(config.TEST.BOX_VOTING_IOU_THRESH, config.TEST.BOX_VOTING_SCORE_THRESH,
                                       with_nms=True)

    with open(test_file) as f:
        image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]

    num_images = len(image_set_index)
    num_classes = len(CLASSES)

    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]


    out_score_list,out_json_list = [],[]

    jsonlistwithscore ,jsonlistwithoutscore = [[] for _ in xrange(num_images)],[[] for _ in xrange(num_images)]

    predict_url(cfg, predictor, image_set_index, nms, box_voting, all_boxes, jsonlistwithscore,
                jsonlistwithoutscore,
                thresh, vis=False, use_box_voting=False, num_gpu=1)


    fout = open(out_prefix + '_vali.txt','w')
    fout_score = open(out_prefix + '_vali_score.txt', 'w')

    for i in range(num_images):
        fout.write(json.dumps(jsonlistwithoutscore[i]) + '\n')
        fout.flush()
        fout_score.write(json.dumps(jsonlistwithscore[i]) + '\n')
        fout_score.flush()


    print("num of images: detection:{}, gt:{}".format(len(all_boxes[0]), num_images))
    #assert len(all_boxes) == num_images, 'calculations not complete'

    # save results
    cache_folder = os.path.join(root_path, 'cache')
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)

    cache_file = os.path.join(cache_folder, dataset + '_' + image_set + '_' + out_prefix + '_detections.pkl')
    with open(cache_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

def load_rfcn_model(cfg,has_rpn,prefix,epoch,ctx):

    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train=False)
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)
    # sym.save('1.json')
    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    num_gpu = len(ctx)
    # infer shape
    SHORT_SIDE = config.SCALES[0][0]
    LONG_SIDE = config.SCALES[0][1]
    DATA_NAMES = ['data', 'im_info']
    LABEL_NAMES = None
    DATA_SHAPES = [('data', (num_gpu, 3, LONG_SIDE, SHORT_SIDE)), ('im_info', (num_gpu, 3))]
    LABEL_SHAPES = None
    data_shape_dict = dict(DATA_SHAPES)
    sym_instance.infer_shape(data_shape_dict)
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    max_data_shape = [[('data', (num_gpu, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    if not has_rpn:
        max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, DATA_NAMES, LABEL_NAMES,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=[DATA_SHAPES], provide_label=[LABEL_SHAPES],
                          arg_params=arg_params, aux_params=aux_params)

    return predictor


def demo_rfcn(cfg, dataset, image_set, root_path, dataset_path,
              idxlist, prefix, epoch, vis, has_rpn, thresh, use_box_voting, test_file, out_prefix):

    predictor_list =[]
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    predictor = load_rfcn_model(cfg,has_rpn,prefix,epoch,ctx)

    demo_net(cfg,predictor_list, dataset, image_set,
             root_path, dataset_path, thresh, vis, use_box_voting, test_file, out_prefix,num_gpu =len(ctx))


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--gpu', help='gpu id', required=True, type=str)
    parser.add_argument('--epoch', help='model epoch', required=True, type=int)
    parser.add_argument('--test_file', help='test file list', required=True, type=str)
    parser.add_argument('--out_prefix', help='output prefix', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--use_box_voting', help='use box voting in test', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))


def main():
    ctx = [mx.gpu(int(args.gpu))]
    print args

    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)

    #arg_params, aux_params = load_param(prefix, epoch, process=False)
    demo_rfcn(config, config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path,
              ctx, os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix), args.epoch,
              args.vis, config.TEST.HAS_RPN, args.thresh, args.use_box_voting, args.test_file, args.out_prefix)


if __name__ == '__main__':
    main()
