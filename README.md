# XAI Evaluation Pipeline
The dataset used in this pipline is : https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

This given repository contains the files used in the pipeline containing three deep learning architectures:
* ResNet50
* EfficientNetB0
* DenseNet121

Explainable AI (XAI) is applied on all of them using four methods
* GradCAM
* GradCAM++
* Integrated Gradients
* SmoothGrad

Then the XAI methods are quantitatively evaluated using
* Overlap : DICE coefficient, Intersection over Union (IoU)
* Faithfulness metrics :
  * Insertion AUC : Pixel Blackout, Pixel Blur, Region Blur
  * Deletion AUC : Pixel Blackout, Pixel Blur, Region Blur
