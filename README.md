# Grocery Dataset 
**Traning set :**
    ```
    generate_label_BI() 
    ```
    Function to generate label for Brand Image
    **X_train** - Brand Images 
    **Y_train** - All pixels are given same class value (corresponding to the brand)

**Test set :** 
    ```
    generate_label_SI(annotation) 
    ```
    Function to generate labels for Shelf Images. Input is dictionary of each shelf image and its corresponding Brand annotation i.e {coordinates and brand number}
    **X_test** - Shelf images 
    **Y_test** - created from Shelf images and BrandImagefromShelf
    Shape obtained from BrandImagesfromShelves for the corresponding Shelf Image and label is created 
    
**Network details :**
    ```
    BUILD_NET_VGG16(vgg16_npy_path=model_path)
    ```
I'm using a pretrained weights og VGG 16 in my **FCN-8** architecture.
Inorder to work on all images sizes I set the **batch_size = 1**. Hence the train on only 1 image returned by ``` Data_Reader()``` function. 
**Total Number of Classes** = 11 {10 Brands + 1 background class}

# Next steps :
[ ] Segmentation images for the test images.
[ ] Bounding Box 
[ ] mAP calculation with (iou >= 0.5) 




