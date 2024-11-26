from PIL import Image
import torch.nn as nn
import copy
from typing import Callable
import utils
import numpy as np
import init_data
import augment_data
import models
import evaluate_data
import compare
                        
import sys


# TODO:
    # Collect results from model into a dataframe or dictionary
    # Properly implement calculating similarities
    # Write paper with hypothesises
    # Implement a Faster RCNN
    # Make sure code can run on Colab
    # Implement VAE data-augmentation
    

def generate_augment_methods(images_and_annotations:[(Image,list[dict])]) ->[(Callable[[Image],Image],Callable[[Image],Image])]:
    
    augment_crop_part = augment_data.create_crop_method(images_and_annotations,is_part=True)
    augment_crop_not_part = augment_data.create_crop_method(images_and_annotations,is_part=False)
    augment_rotate_part = augment_data.create_rotate_method()
    augment_rotate_not_part = augment_data.create_rotate_method()
    augment_flip_part = augment_data.create_flip_method()
    augment_flip_not_part = augment_data.create_flip_method()
    
    augment_methods = [(augment_crop_part,augment_crop_not_part),
                       (augment_rotate_part,augment_rotate_not_part),
                       (augment_flip_part,augment_flip_not_part)]
    return augment_methods

'''
Include any parts of a test that include randomness such as:
- Generating the initial train set
- Augmenting the initial train set
- Generating the not part test set
- Evaluating the model
'''
def run_trial(augment_method_pair:(Callable[[Image],Image],Callable[[Image],Image]),
              train_images_and_annotations:[(Image,list[dict])],
              test_images_and_annotations:[(Image,list[dict])],
              model:nn.Module,
              part_list_size:int=10,
              init_not_part_ratio=1.0,
              full_not_part_ratio=1.0,
              num_train_epochs:int=10,
              verbose:bool=True,
              save_images:int=0):
    part_augment_method, not_part_augment_method = augment_method_pair
    train_init_part_list, train_init_not_part_list = init_data.create_initial_lists(train_images_and_annotations,
                                                                                    init_not_part_ratio=init_not_part_ratio)
    train_full_part_list, train_full_not_part_list = augment_data.augment(train_init_part_list,
                                                                          train_init_not_part_list,
                                                                          part_augment_method=part_augment_method,
                                                                          not_part_augment_method=not_part_augment_method,
                                                                          part_list_size=part_list_size,
                                                                          full_not_part_ratio=full_not_part_ratio)
    test_init_part_list, test_init_not_part_list = init_data.create_initial_lists(test_images_and_annotations,
                                                                                  init_not_part_ratio=1.0)
    if verbose:
        print(f"Parts ({len(train_init_part_list)})")
        utils.view_sample_images_from_list(train_init_part_list,num_samples=20)
        print(f"Not Parts ({len(train_init_not_part_list)})")
        utils.view_sample_images_from_list(test_init_not_part_list,num_samples=20)
    if save_images > 0:
        utils.save_test_images_to_folder(f"{save_images}-not-part",train_full_not_part_list)
        utils.save_test_images_to_folder(f"{save_images}-part",train_full_part_list)
        
    result, wrong, right = evaluate_data.evaluate(model,
                                                  (train_full_part_list,train_full_not_part_list),
                                                  (test_init_part_list,test_init_not_part_list),
                                                  num_epochs=num_train_epochs)
    if verbose:
        print(f"Accuracy: {result}")
        utils.view_sample_labeled_images_from_list(wrong,num_samples=10)
        utils.view_sample_labeled_images_from_list(right,num_samples=10)
        
    if save_images > 0:
        utils.save_test_images_to_folder(f"{save_images}-wrong",[utils.get_image(t) for t in wrong])
        utils.save_test_images_to_folder(f"{save_images}-right",[utils.get_image(t) for t in right])
    
    # Compare wrong, right 
    comparison_data = {compare.TRAIN_PART_KEY:train_full_part_list,
                       compare.TRAIN_NOT_PART_KEY:train_full_not_part_list,
                       compare.TEST_PART_KEY:test_init_part_list,
                       compare.TEST_NOT_PART_KEY:test_init_not_part_list,
                       compare.TRAIN_IMAGES_KEY: [img for img, annotations in train_images_and_annotations],
                       compare.TEST_IMAGES_KEY: [img for img, annotations in test_images_and_annotations]}
        
    comparisons = compare.run_comparison(comparison_data)
    
    if verbose:
        compare.view_comparison_results(comparisons)
    
    return result, wrong, right, comparisons
        
        

def main():
    assert utils.get_int_from_ratio(1.2,10) == 12
    test_image = Image.open("../potential_dataset/NASA/easy diagram 1.png")#easy diagram 1.png
    test_annotations = [{'x1': 97, 'y1': 16, 'x2': 243, 'y2': 71}, 
                        {'x1': 99, 'y1': 97, 'x2': 241, 'y2': 150}, 
                        {'x1': 97, 'y1': 177, 'x2': 241, 'y2': 231}, 
                        {'x1': 294, 'y1': 181, 'x2': 425, 'y2': 232}, 
                        {'x1': 296, 'y1': 101, 'x2': 425, 'y2': 150}, 
                        {'x1': 36, 'y1': 274, 'x2': 86, 'y2': 324}, 
                        {'x1': 97, 'y1': 316, 'x2': 240, 'y2': 363}, 
                        {'x1': 56, 'y1': 356, 'x2': 96, 'y2': 391}, 
                        {'x1': 293, 'y1': 357, 'x2': 399, 'y2': 410}, 
                        {'x1': 97, 'y1': 416, 'x2': 240, 'y2': 465}, 
                        {'x1': 290, 'y1': 437, 'x2': 439,'y2': 485}, 
                        {'x1': 95, 'y1': 493, 'x2': 238, 'y2': 544}, 
                        {'x1': 95, 'y1': 576, 'x2': 238, 'y2': 624}, 
                        {'x1': 293, 'y1': 538, 'x2': 439, 'y2': 585}]
    
    test_image_2 = Image.open("../potential_dataset/NASA/easy diagram 2.jpg")
    test_annotations_2 = [{'x1': 331, 'y1': 273, 'x2': 398, 'y2': 310},
                          {'x1': 383, 'y1': 212, 'x2': 424, 'y2': 248},
                          {'x1': 265, 'y1': 175, 'x2': 309, 'y2': 208},
                          {'x1': 196, 'y1': 143, 'x2': 257, 'y2': 239}, 
                          {'x1': 549, 'y1': 202, 'x2': 628, 'y2': 247},
                          {'x1': 565, 'y1': 246, 'x2': 617, 'y2': 307}, 
                          {'x1': 163, 'y1': 667, 'x2': 190, 'y2': 710}, 
                          {'x1': 196, 'y1': 645, 'x2': 222, 'y2': 734},
                          {'x1': 160, 'y1': 355, 'x2': 221, 'y2': 416}, 
                          {'x1': 160, 'y1': 553, 'x2': 222, 'y2': 613}, 
                          {'x1': 554, 'y1': 346, 'x2': 625, 'y2': 412},
                          {'x1': 746, 'y1': 594, 'x2': 802, 'y2': 640}, 
                          {'x1': 671, 'y1': 452, 'x2': 718, 'y2': 495}, 
                          {'x1': 674, 'y1': 385, 'x2': 717, 'y2': 424}, 
                          {'x1': 743, 'y1': 386, 'x2': 789, 'y2': 424}, 
                          {'x1': 745, 'y1': 460, 'x2': 783, 'y2': 495}, 
                          {'x1': 819, 'y1': 426, 'x2': 872, 'y2': 456},
                          {'x1': 435, 'y1': 163, 'x2': 504, 'y2': 206}, 
                          {'x1': 731, 'y1': 209, 'x2': 797, 'y2': 242},
                          {'x1': 884, 'y1': 272, 'x2': 908, 'y2': 330}, 
                          {'x1': 923, 'y1': 299, 'x2': 1109, 'y2': 620}, 
                          {'x1': 134, 'y1': 111, 'x2': 253, 'y2': 759}]
    
    test_image_3 = Image.open("../potential_dataset/NASA/easy diagram 3.jpg")
    test_annotations_3 = [{'x1': 201, 'y1': 74, 'x2': 379, 'y2': 204}, 
                          {'x1': 204, 'y1': 211, 'x2': 398, 'y2': 324},
                          {'x1': 434, 'y1': 208, 'x2': 596, 'y2': 327}, 
                          {'x1': 630, 'y1': 209, 'x2': 807, 'y2': 323}, 
                          {'x1': 831, 'y1': 210, 'x2': 973, 'y2': 328},
                          {'x1': 841, 'y1': 362, 'x2': 973, 'y2': 477}, 
                          {'x1': 1000, 'y1': 367, 'x2': 1128, 'y2': 477}]
    
    
    train_images_and_annotations = [(test_image,test_annotations),
                                    (test_image_2,test_annotations_2)]
    
    test_images_and_annotations = [(test_image_3,test_annotations_3)]
    
    augment_methods = generate_augment_methods(train_images_and_annotations)
    model_list = [("Simple CNN",models.build_model()),
                  ("Pretrained CNN",models.load_model("test-from-pretrain_model-main.pth"))]
    NUM_ITERATIONS = 3
    
    result,wrong,right,comparisons = run_trial(augment_methods[0],
                                               train_images_and_annotations,
                                               test_images_and_annotations,
                                               model_list[1][1],
                                               part_list_size=10,
                                               init_not_part_ratio=1.0,
                                               full_not_part_ratio=1.0,
                                               num_train_epochs=1,
                                               verbose=True,
                                               save_images=1)
    
    
#     for augment_method_pair in augment_methods:
#         for model_name, model in model_list:
#             accuracies = []
#             for iteration in range(0,NUM_ITERATIONS):
#                 iter_model = utils.copy_model(model)
#                 result,wrong,right = run_trial(augment_method_pair,
#                                                train_images_and_annotations,
#                                                test_images_and_annotations,
#                                                iter_model,
#                                                part_list_size=10,
#                                                init_not_part_ratio=1.0,
#                                                full_not_part_ratio=1.0,
#                                                num_train_epochs=10,
#                                                verbose=True)
#                 accuracies.append(result)
#             print(f"Augment: ({augment_method_pair[0].__name__},{augment_method_pair[1].__name__}) model: {model_name}")
#             print(f"Avg Accuracy:{np.mean(accuracies):.2f} +/- {np.std(accuracies):.2f}")
            
    
    
    

    
if __name__ == "__main__":
    sys.exit(main())
    
    
'''
Areas for future research:
- Smart sampling an image instead of being completely random

'''