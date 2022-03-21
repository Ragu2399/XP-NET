# XP-NET
An Attention Segmentation Network by Dual Teacher Hierarchical Knowledge distillation for Polyp Generalization

Endoscopic imaging is largely used as the diagnostic tool
for Colon polyps-induced GI tract cancer. This diagnosis via
image identification requires a high amount of experience
that may be lacking in untrained physicians. Hence, using a
software-aided approach to detect those anomalies may better
identify the tissue abnormalities. In this paper, a novel deep
learning network ’XP-Net’ with Effective Pyramidal Squeeze
Attention (EPSA) module using Hierarchical Adversar-
ial Knowledge Distillation by a combination of
two teacher networks is proposed. Its adds ‘complementary
knowledge’ to the student network– thus aiding in the im-
provement of network performance. The lightweight EPSA
block enhanced the current network architecture by capturing
multi-scale spatial information of objects at a granular level
with long-range channel dependency. The XP-Net compiled
into the NVIDIA TensorRT engine gave a better real-time per-
formance in terms of throughput. The proposed network has
achieved a dice score of 0.839 and IoU of 0.805 in the valida-
tion data set, and it was able to attain an average throughput
of 60 fps in mobile GPU. This proposed deep learning-based
segmentation approach is expected to aid clinicians in ad-
dressing the complications involved in the identification and
removal of precancerous anomalies more competently.


The model was trained for ENDOCV2022 challenge.

Pretrained weights 

https://drive.google.com/drive/folders/1c7w_DYhcIjBzoHqHVTdbNTa2ObvXCBUf?usp=sharing
