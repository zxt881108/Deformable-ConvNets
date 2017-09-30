## 暴恐 + online hard example mining训练流程
1. 在mxnet环境的容器中clone如下代码。这个是Deformable convnets的代码，不过其中有faster rcnn/frcn的ohem实现，可以直接拿过来用。  
`git clone https://github.com/ataraxialab/Deformable-ConvNets.git`
2. 重新编译mxnet    
`cp -r $(DCN_ROOT)/rfcn/operator_cxx/* $(MXNET_ROOT)/src/operator/contrib/`     
`cd ${MXNET_ROOT}`
`make -j $(nproc)`
3. 准备数据和预训练模型：将暴恐数据放在`./data/terror/`目录下，将预训练的resnet101模型（下载地址：`https://1drv.ms/u/s!Am-5JzdW2XHzhqMSjehIcCgAhvEAHw`）放在`./model/pretrained_model/resnet_v1_101-0000.params`
4. 更改配置：我们使用rfcn框架进行训练，首先更改配置文件`./experiments/rfcn/cfgs/resnet_v1_101_terror_rfcn_end2end_ohem.yaml`，具体的，包括gpu数量，数据存放位置的修改，模型保存名称的修改，ENABLE_OHEM置为true等。
5. 开始训练：
`python experiments/rfcn/rfcn_end2end_train_test.py --cfg experiments/rfcn/cfgs/resnet_v1_101_terror_rfcn_end2end_ohem.yaml`
