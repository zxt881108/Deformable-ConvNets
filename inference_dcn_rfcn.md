## 暴恐 + 推理流程
1. 在mxnet环境的容器中clone如下代码。  
`git clone https://github.com/ataraxialab/Deformable-ConvNets.git`
设置`${DCN_ROOT}` 和 `${MXNET_ROOT}`路径
2. 重新编译mxnet    
`cd ${DCN_ROOT}`
`sh ./init.sh`
`cp -r ${DCN_ROOT}/rfcn/operator_cxx/* ${MXNET_ROOT}/src/operator/contrib/`     
`cd ${MXNET_ROOT}`
`make clean`
`make -j $(nproc) USE_OPENCV=1 USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1`
3. 准备训练模型: 将训练的resnet101模型（下载地址：`http://otr41gcz3.bkt.clouddn.com/weiboneg.tar`）解压到 `$(DCN_ROOT)/output/rfcn_dcn-terror/resnet_v1_101_terror_dcn_test_rfcn_end2end_ohem/2007_trainval/`
4. 更改配置：更改配置文件`$(DCN_ROOT)/experiments/rfcn/cfgs/resnet_v1_101_terror_dcn_test_rfcn_end2end_ohem.yaml`，具体的，`test_epoch` 改为8 `BATCH_IMAGES` 改为1 `model_prefix` 改为 `rfcn_voc`
5. 在 `$(DCN_ROOT)` 建立 `test.txt` 文件，每一行为一个图片链接
6. 开始批量推理：
`python experiments/rfcn/rfcn_terror.py --cfg experiments/rfcn/cfgs/resnet_v1_101_terror_dcn_test_rfcn_end2end_ohem.yaml`
