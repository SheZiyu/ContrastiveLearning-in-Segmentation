# ContrastLearning-in-Segmentation

This is my practice for Grand Challenge AMOS 2022, which is about the challenging task multi-organ (15 abdominal organs) segmentation. https://amos22.grand-challenge.org/

The code is derived from the following papers and GitHub repositories:


![image](https://github.com/SheZiyu/ContrastLearning-in-Segmentation/assets/98766434/2cd0d575-b27b-4e4d-a633-2ab9ad7930a2)

Paper1: Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International conference on machine learning. PMLR, 2020. https://arxiv.org/abs/2002.05709

Paper2: Zeng, Dewen, et al. "Positional contrastive learning for volumetric medical image segmentation." Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part II 24. Springer International Publishing, 2021. https://miccai2021.org/openaccess/paperlinks/2021/09/01/372-Paper1432.html

GitHub Repository1: https://github.com/google-research/simclr

GitHub Repository2: https://github.com/dewenzeng/positional_cl

I used 2D UNet to deal with 2D slices derduring traing time and do the extended the 2D network into the 3D network to deal with Overall Survival (OS) classification of patients with Primary Central Neural System Lymphoma (PSNSL) using 3D Magnetic Resonance Imaging (MRI). 

 
Sorry for bother you. I was trying to modify the segmentation model. Our first goal is to segment different organs in the abdominal area.
 
For the input, the original images are 3D images. 
For the model, I choose 2D UNET to deal with the 2D slices and then during the inference time, I splice all the 2d slices together to a 3D volume.
For the loss, I choose the similar loss as the contrastive loss in the SimCLR paper.
 
But as you can see, the alignment of the top images (original images) and the bottom images (segmentation results of the model) is not good, seems the segmentation is zoomed out by the model. 
 
Do you have any idea to improve it?
 
Thanks so much!

