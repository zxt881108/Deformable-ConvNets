# add light head in dcn rfcn

## 参考文献
    [Light-Head R-CNN: In Defense of Two-Stage Object Detector](https://export.arxiv.org/pdf/1711.07264)

## 流程图
![](add_light_head.png)

### 具体实现
流程图中下半部分为light-head内容。
具体包括

1. 大卷积核的split 卷积
	使用 [global_context](https://github.com/ElaineBao/mxnet/blob/master/example/rcnn/rcnn/symbol/roi_global_context.py) 代码实现。
2. 将两个纬度的卷积做一个合并
	[code](https://github.com/ataraxialab/Deformable-ConvNets/blob/dev-global-context/rfcn/symbols/resnet_v1_101_rfcn_dcn.py#L817-L821) 
4. 利用FC层达到输出纬度的要求
	[code](https://github.com/ataraxialab/Deformable-ConvNets/blob/dev-global-context/rfcn/symbols/resnet_v1_101_rfcn_dcn.py#L822-L823)

 


