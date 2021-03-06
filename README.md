# MMDetection-v1.1.0
Object detection experiments based on [MMDetection: Open MMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection). 

**Setup**: <a href='https://github.com/v-qjqs/mmdetection-v1.1.0/blob/master/mmdetection/docs/INSTALL.md'>Installation</a><br>

**Environments**:
* Hardware: Ubuntu 16.04.6, 8 GeForce RTX 2080 Ti GPU
* Software: Python 3.5.2, PyTorch 1.3.1, CUDA 10.0.130, CUDNN 7.5.0


## Major Modification/Adding Compared with [MMDetection](https://github.com/open-mmlab/mmdetection)
* Adding [implementation](mmdetection/mmdet/models/backbones/DetNASNet/) of [DetNASNet](https://arxiv.org/pdf/1903.10979.pdf) as backbone. [NIPS 2019 Paper: DetNAS: Backbone Search for Object Detection.](https://arxiv.org/pdf/1903.10979.pdf) DetNasNet is got by adopting [Single Path One-Shot](https://arxiv.org/abs/1904.00420) Nearal Architecture Search (NAS) under [ShuffleNetV2](https://arxiv.org/abs/1807.11164)-like search space. Object Detection experiments with [FPN](https://arxiv.org/abs/1612.03144) on COCO show that, DetNasNet achieves promising enhancement (37.3->42.0, +4.7% mAP) than Res-50 backbone under comparable FLOPS (3.8G).
* Adding [NaiveSyncBatchNorm](mmdetection/mmdet/models/utils/norm.py#L57) refferred from the implementation of [Detectron2](https://github.com/facebookresearch/detectron2).


## Detection/Instance Segmentation Performance on [COCO17 Val Dataset](http://cocodataset.org/index.htm#download)
### Anchor-based Detector:
#### RetinaNet-Res50-FPN
model | backbone | Lr schd | [box mAP](http://cocodataset.org/index.htm#detection-eval) | config
------------- | ------------- | ------------- | ------------- | -------------
**BASELINE** (bn, frozen statistics) | res50 | 1x | 35.4, report:35.6  | retinanet_r50_fpn_1x
bn -> torch_syncbn | res50 | 1x | 35.5 | retinanet_r50_fpn_1x_torch_syncbn
bn -> detectron2_syncbn | res50 | 1x | 35.3 | retinanet_r50_fpn_1x_detectron2_syncbn
bn -> [gn](https://arxiv.org/abs/1803.08494) | res50 | 1x | 35.4 | retinanet_r50_fpn_1x_gn
nms -> [soft_nms](https://arxiv.org/abs/1704.04503)| res50 | 1x | 35.5  | retinanet_r50_fpn_1x_softnms
|+ [guided_anchoring](https://arxiv.org/abs/1901.03278)| res50 | 1x | 35.6  | ga_retinanet_r50_fpn_1x
|+ [DCN](https://arxiv.org/abs/1703.06211) | res50 | 1x | 38.8 | retinanet_r50_fpn_1x_dconv_c3-c5
|+ [DCNV2](https://arxiv.org/abs/1811.11168) | res50 | 1x | 39.1 | retinanet_r50_fpn_1x_mdconv_c3-c5_bn
|+ [libra-rcnn](https://arxiv.org/abs/1904.02701) | res50 | 1x | 37.4, report: 37.7 | libra_retinanet_r50_fpn_1x
|+ [gcnet](https://arxiv.org/abs/1904.11492) | res50 | 1x | 37.6 | retinanet_r50_fpn_1x (gcb: r4)
|+ [DCNV2](https://arxiv.org/abs/1811.11168) + [gcnet](https://arxiv.org/abs/1904.11492) + [libra](https://arxiv.org/abs/1904.02701)  | res50 | 1x | **41.0** | retinanet_r50_fpn_1x_mdconv_c3-c5_gcnet_c3-c5_libra (gcb: r4)

#### MaskRCNN-Res50-FPN
model | backbone | Lr schd | box AP | mask AP | config
------------- | ------------- | ------------- | ------------- | ------------- | -------------
**BASELINE** (bn, frozen statistics) | res50 | 1x | 37.3, report: 37.4 | 34.2, report: 34.3 | mask_rcnn_r50_fpn_1x
bn -> [gn](https://arxiv.org/abs/1803.08494) | res50 | 1x | 37.1 | 33.9 | MY/mask_rcnn_r50_fpn_1x_gn_notall 
bn -> [gn](https://arxiv.org/abs/1803.08494), + gn_for_head | res50 | 1x | 37.2 | 34.4 | MY/mask_rcnn_r50_fpn_1x_gn (all)
bn -> torch_syncbn | res50 | 1x | 37.2 | 33.9 | MY/mask_rcnn_r50_fpn_1x_syncbn
bn -> detectron2_syncbn | res50 | 1x | 37.4 | 34.1 | MY/mask_rcnn_r50_fpn_1x_detectron2_syncbn
+[gcnet](https://arxiv.org/abs/1904.11492) | res50 | 1x | 38.8, report:38.9 | 35.4, report:35.5 | gcnet/mask_rcnn_r4_gcb_c3-c5_r50_fpn_1x
+[gcnet](https://arxiv.org/abs/1904.11492), bn -> torch_syncbn | res50 | 1x | 39.6, report:39.9 | 36.0, report:36.2 | gcnet/mask_rcnn_r4_gcb_c3-c5_r50_fpn_syncbn_1x
+[gcnet](https://arxiv.org/abs/1904.11492), bn -> detectron2_syncbn | res50 | 1x | 39.9 | 36.1 | gcnet/mask_rcnn_r4_gcb_c3-c5_r50_fpn_detectron2_syncbn_1x
+[libra](https://arxiv.org/abs/1904.02701) | res50 | 1x | 39.2 | 35.3 | gcnet/retinanet_r50_fpn_1x_MY
+[DCN](https://arxiv.org/abs/1703.06211) | res50 | 1x | 41.2, report:41.1 | 37.3, report:37.2 | mask_rcnn_dconv_c3-c5_r50_fpn_1x
+[DCNV2](https://arxiv.org/abs/1811.11168) | res50 | 1x | 41.0, report:41.3 | 37.1, report:37.3 | mask_rcnn_mdconv_c3-c5_r50_fpn_1x
+[DCNV2](https://arxiv.org/abs/1811.11168)+[gcnet](https://arxiv.org/abs/1904.11492) | res50 | 1x | 42.0 | 37.9 | mask_rcnn_r50_fpn_1x_mdconv_gcb_c3-c5
+[DCNV2](https://arxiv.org/abs/1811.11168)+[libra](https://arxiv.org/abs/1904.02701)  | res50 | 1x | 42.6 | 37.9 | mask_rcnn_r50_fpn_1x_mdconv_c3-c5_libra
+[DCNV2](https://arxiv.org/abs/1811.11168)+[gcnet](https://arxiv.org/abs/1904.11492)+[libra](https://arxiv.org/abs/1904.02701)  | res50 | 1x | 43.1 | 38.2 | mask_rcnn_r50_fpn_1x_mdconv_gcb_c3-c5_libra
+[DCNV2](https://arxiv.org/abs/1811.11168)+[gcnet](https://arxiv.org/abs/1904.11492)+[libra](https://arxiv.org/abs/1904.02701), bn -> detectron2_syncbn | res50 | 1x | 43.6, +flip:44.6 | 38.6, +flip:39.2 | mask_rcnn_r50_fpn_1x_mdconv_gcb_c3-c5_libra_detectron2_syncbn
+[DCNV2](https://arxiv.org/abs/1811.11168)+[gcnet](https://arxiv.org/abs/1904.11492)+[libra](https://arxiv.org/abs/1904.02701)+ms_train, bn -> detectron2_syncbn | res50 | 1x | **44.2**, +flip:**44.9** | **38.9**, +flip:**39.4** | mask_rcnn_r50_fpn_1x_mdconv_gcb_c3-c5_libra_detectron2_syncbn_mt ?


#### MaskRCNN-DetNASNet-FPN
model | backbone | FLOPs  | Lr schd | box AP | mask AP | config
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
maskrcnn_r50_fpn  | res50 | 3.8G| 1x | 37.4 | 34.1 | MY/mask_rcnn_r50_fpn_1x_detectron2_syncbn
maskrcnn_r50_fpn, **BASELINE** | [DetNasNet](https://arxiv.org/pdf/1903.10979.pdf) | 3.8G| 1x | 33.1 | 30.0 | detnasnet_detectron2_syncbn/mask_rcnn_fpn_1x
|+ [gcnet](https://arxiv.org/abs/1904.11492) | [DetNasNet](https://arxiv.org/pdf/1903.10979.pdf) | 3.8G| 1x | 34.5 | 31.2 | detnasnet_detectron2_syncbn/mask_rcnn_gcb_c4-c5_fpn_1x
|+ [gcnet](https://arxiv.org/abs/1904.11492)+[libra](https://arxiv.org/abs/1904.02701) | [DetNasNet](https://arxiv.org/pdf/1903.10979.pdf) | 3.8G| 1x | 35.0 | 30.9 | detnasnet_detectron2_syncbn/mask_rcnn_gcb_c4-c5_libra_fpn_1x

#### Cascade-MaskRCNN-Res50-FPN
model | backbone | Lr schd | box AP | mask AP | config
------------- | ------------- | ------------- | ------------- | ------------- | -------------
cascade_maskrcnn, **BASELINE** | res50 | 1x | 41.2, report: 41.2 | 35.7, report: 35.7| cascade_mask_rcnn_r50_fpn_1x
+[DCNV2](https://arxiv.org/abs/1811.11168)+[gcnet](https://arxiv.org/abs/1904.11492)+[libra](https://arxiv.org/abs/1904.02701), bn -> detectron2_syncbn | res50 | 1x | 46.0, +flip:46.9 | 39.4, +flip:39.9 | cascade_mask_rcnn_r50_mdconv_gcb_libra_detectron2_syncbn_fpn_1x_MY
+[DCNV2](https://arxiv.org/abs/1811.11168)+[gcnet](https://arxiv.org/abs/1904.11492)+[libra](https://arxiv.org/abs/1904.02701)+ms_train, bn -> detectron2_syncbn | res50 | 1x | **46.8**, +flip:**47.3** | **39.9**, +flip:**40.3** | cascade_mask_rcnn_r50_mdconv_gcb_libra_detectron2_syncbn_fpn_1x_mt_MY
+[DCNV2](https://arxiv.org/abs/1811.11168)+[gcnet](https://arxiv.org/abs/1904.11492)+[libra](https://arxiv.org/abs/1904.02701)+ms_train, bn -> detectron2_syncbn | res50 | 20e | +flip:**48.3** | +flip:**40.9** | cascade_mask_rcnn_r50_mdconv_gcb_libra_detectron2_syncbn_fpn_20e_1x_mt_MY


#### Hybrid Task Cascade (HTC)
model | backbone | Lr schd | box AP | mask AP | config
------------- | ------------- | ------------- | ------------- | ------------- | -------------
[Hybrid Task Cascade (HTC)](https://arxiv.org/abs/1901.07518), **BASELINE** | res50 | 1x | 41.5 | 36.5 | htc/htc_without_semantic_r50_fpn_1x
+[DCNV2](https://arxiv.org/abs/1811.11168)+[gcnet](https://arxiv.org/abs/1904.11492)+[libra](https://arxiv.org/abs/1904.02701)+ms_train, bn -> detectron2_syncbn | res50 | 1x | **47.1**, +flip:**47.8** | **40.7**, +flip:**41.2** | htc/htc_without_semantic_r50_fpn_mdconv_gcb_libra_detectron2_syncbn_mt_1x_my
+[DCNV2](https://arxiv.org/abs/1811.11168)+[gcnet](https://arxiv.org/abs/1904.11492)+[libra](https://arxiv.org/abs/1904.02701)+ms_train, bn -> detectron2_syncbn | res50 | 20e | +flip:**48.8** | +flip:**42.0** | htc/htc_without_semantic_r50_fpn_20e_mdconv_gcb_libra_detectron2_syncbn_mt_1x_my


### Anchor-free based Detector:
model | backbone | Lr schd | box AP  | config
------------- | ------------- | ------------- | ------------- | -------------
reppoints, **BASELINE** (bn, frozen statistics) | res50 | 1x | 36.8, reported:36.8 | reppoints_moment_r50_no_gn_fpn_1x
reppoints | res50 | 1x | 37.9, report:38.2 | reppoints_moment_r50_fpn_1x (+gn_neck_head)
reppoints, bn -> detectron2_syncbn | res50 | 1x | ? | (+gn_neck_head)
reppoints+mdconv+gcb+libra+ms_train| res50 | 1x | 42.1, +flip: 42.4 | reppoints_moment_r50_dcn_gcb_libra_fpn_1x_mt_MY (+gn_neck_head) 
reppoints+mdconv+gcb+libra+ms_train, bn -> detectron2_syncbn | res50 | 1x | **43.1**, +flip:**43.4** | reppoints_moment_r50_dcn_gcb_libra_fpn_detectron2_syncbn_1x_mt_MY (+gn_neck_head) 
fcos, **BASELINE** | res50 | 1x | 35.6 | fcos_r50_fpn_gnhead_1x_my
fcos+dcn+gcb+libra | res50 | 1x | 39.5 | fcos_r50_fpn_mdconv_gcb_c3-c5_libra_gnhead_1x
fcos+dcn+gcb+libra+ms_train | res50 | 1x | 39.7(no flip) | fcos_r50_fpn_mdconv_gcb_c3-c5_libra_gnhead_1x_mt
[guided_anchoring](https://arxiv.org/abs/1901.03278), **BASELINE** | res50 | 1x | 35.6  | ga_retinanet_r50_fpn_1x_MY
+dcn+gcb+libra+ms_train, bn -> detectron2_syncbn | res50 | 1x | **41.8**  | ga_retinanet_r50_mdconv_gcb_libra_fpn_detectron2_syncbn_1x_mt_MY
+dcn+gcb+libra+ms_train+gn_for_neck_head, bn -> detectron2_syncbn | res50 | 1x | **41.7**  | ga_retinanet_r50_mdconv_gcb_libra_fpn_detectron2_syncbn_gnneckhead_1x_mt_MY


