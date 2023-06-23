# ContrastiveLearning-in-Segmentation

This is my practice for Grand Challenge AMOS 2022, which is about the challenging task of abdominal multi-organ (15 abdominal organs) segmentation. https://amos22.grand-challenge.org/

The code is derived from the following papers and GitHub repositories:

![image](https://github.com/SheZiyu/ContrastLearning-in-Segmentation/assets/98766434/2cd0d575-b27b-4e4d-a633-2ab9ad7930a2)

Paper1: Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International conference on machine learning. PMLR, 2020. https://arxiv.org/abs/2002.05709

Paper2: Zeng, Dewen, et al. "Positional contrastive learning for volumetric medical image segmentation." Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part II 24. Springer International Publishing, 2021. https://miccai2021.org/openaccess/paperlinks/2021/09/01/372-Paper1432.html

GitHub Repository1: https://github.com/google-research/simclr

GitHub Repository2: https://github.com/dewenzeng/positional_cl

I used 2D UNet to deal with 2D input slicing from original 3D images during training, then I spliced the corresponding 2D slices together to a 3D volume for each 3D image during inference. For loss function, I chose the data augmentation contrastive loss (Paper1) and the positional contrastive loss (Paper2).

Qualitative Result: The top image is the original 3D image viewing from x, y and z-axis. The bottom image is the corresponding segmentation result of the model.

https://github.com/SheZiyu/ContrastLearning-in-Segmentation/assets/98766434/15dc3c22-0cd8-4da7-a71f-19d63fa11cbe

 
 
 



