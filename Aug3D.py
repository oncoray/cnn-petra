import numpy as np 
import tensorflow as tf
#imports for data augmentations 
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform, BrightnessTransform,GammaTransform,BrightnessMultiplicativeTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform,SpatialTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.noise_transforms import RicianNoiseTransform,GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.abstract_transforms import RndTransform
from batchgenerators.dataloading.data_loader import DataLoaderBase, SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform


'''
3D Data Augmentation using batchgenerators library 
Data is augmented on fly using tf data_loader function 
Several types of data augmentation is possible, further deatils can be found on github --> https://github.com/MIC-DKFZ/batchgenerators

'''
'''Helper function for data augmentation'''
class AugmentDataLoader(SlimDataLoaderBase):
    def __init__(self, data, BATCH_SIZE=1):
        super(AugmentDataLoader, self).__init__(data, BATCH_SIZE) 
        # data is now stored in self._data.
    
    def generate_train_batch(self):
        return {'data':self._data[None,None].astype(np.float32), 'some_other_key':'some other value'}



@tf.function
def augment_data(volume):
    def apply_augment(volume):
#        my_transforms = []
        
#         brightness_transform = BrightnessTransform(mu=0.1, sigma=0.05)
#         brightness_transform_fifty = RndTransform(brightness_transform,prob=0.5)
#         my_transforms.append(brightness_transform_fifty)
        
#         Contrast_transform = ContrastAugmentationTransform((0.3, 1.75), preserve_range=True)
#         Contrast_transform_fifty = RndTransform(Contrast_transform,prob=0.7)
#         my_transforms.append(Contrast_transform_fifty)
        
#         spatial_transform = SpatialTransform((60,60,44),(60,60,44),
#             do_elastic_deform=True, alpha=(0.0, 150.0), sigma=(8.0, 13.0),
#                                              do_rotation=False,
#                                              do_scale=False,
#                                              border_mode_data='nearest', border_cval_data=0, order_data=0,
#                                              random_crop=False)
#         spatial_transform_fifty = RndTransform(spatial_transform, prob=0.2)
#         my_transforms.append(spatial_transform_fifty)
#        all_transforms = Compose(my_transforms)
######################################################Brats transformations ##################################
        tr_transforms = []
    
        #tr_transforms.append(ContrastAugmentationTransform((1.0, 1.75), per_channel=True, p_per_sample=0.15))
    
        #tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
        
        tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
            # we can also invert the image, apply the transform and then invert back
        tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

            # Gaussian Noise
        tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

            # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
            # thus make the model more robust to it
        tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                                       p_per_channel=0.5, p_per_sample=0.15))
        patch_size = (60,60,44)

#         tr_transforms.append(
#                         SpatialTransform_2(
#                             patch_size, do_elastic_deform=False,
#                             do_rotation=True,
#                             angle_x= (0,0),
#                             angle_y=(0,0),
#                             angle_z=(-10 / 360. * 2 * np.pi, 10 / 360. * 2 * np.pi),
#                             do_scale=False, random_crop=False,
#                             border_mode_data='constant', border_cval_data=0,
#                             p_el_per_sample=0.00, p_rot_per_sample=0.15, p_scale_per_sample=0.00
#                         )
#                     )

            # now we compose these transforms together
        all_transforms = Compose(tr_transforms)
###################################################################################################################################

        multithreaded_generator = SingleThreadedAugmenter(AugmentDataLoader(volume), all_transforms)
        volume = multithreaded_generator.next()['data']
        return volume

    augmented_volume = tf.numpy_function(apply_augment, [volume], tf.float32)
    return augmented_volume