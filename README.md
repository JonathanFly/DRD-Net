# DRD-Net

# For Test the derain method: please download the pre-trained model from https://drive.google.com/open?id=17njD0WTNKluoZmie6apCWn6jgJSOgqaF
a) Test the Rain200H dataset: Use the Rain200H_test.py to test the rain200H

b) Test the Rain200L dataset: Use the Rain200L_test.py to test the rain200L

c) Test the Rain800 dataset: Use the Rain800_test.py to test the rain800


# For Train the model, you can download the rain image dataset.
a) put the rainy/clean in two different catalogue (make sure that the corresponding rainy/clean pair images have the same name).
-- image_dir_noise(rainy train image dir); --image_dir_original(clean train image dir); --test_dir_noise(rainy test image dir); --test_dir_original(clean test image dir) 

b) You can choose the --If_n as True to normalize the image, or set the --If_n as False not to normalize the image.
