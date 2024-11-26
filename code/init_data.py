from PIL import Image
import torch.nn as nn
import copy
from typing import Callable
import utils
import init_data
import augment_data

# In: image, annotations, augment method,
# Out: part list, not_part_list

def create_initial_lists(images_and_annotations:[(Image,list[dict])],
                         init_not_part_ratio:float=1.0) -> (list[Image],list[Image]):
    # crop annotations
    init_part_list = []
    init_not_part_list = []
    for image, annotation in images_and_annotations:
        sub_part_list, sub_not_part_list = create_initial_lists_from_image(image,annotation,init_not_part_ratio)
        init_part_list.extend(sub_part_list)
        init_not_part_list.extend(sub_not_part_list)
    return init_part_list, init_not_part_list
    
def create_initial_lists_from_image(image:Image,
                                    annotations:list[dict],
                                    init_not_part_ratio:float=1.0) -> (list[Image],list[Image]):
    parts_list = utils.crop_annotations_from_image(image,annotations)
    init_not_parts_list_size = utils.get_int_from_ratio(init_not_part_ratio,len(parts_list))
    not_parts_list = utils.generate_not_part_images(image,annotations,init_not_parts_list_size)
    return parts_list, not_parts_list

                                       
import sys

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
    
    #test_image.show()
    images_and_annotations = [(test_image,test_annotations),
                              (test_image_2,test_annotations_2),
                              (test_image_3,test_annotations_3)]
    
    part_list, not_part_list = create_initial_lists(images_and_annotations,init_not_part_ratio = 0.5)
    assert len(not_part_list) < len(part_list)
    part_list, not_part_list = create_initial_lists(images_and_annotations,init_not_part_ratio = 1.5)
    assert len(not_part_list) > len(part_list)
    part_list, not_part_list = create_initial_lists(images_and_annotations,init_not_part_ratio = 1.0)
    assert len(not_part_list) == len(part_list)
    
    SAVE_FOLDER = "../images/test_image_sets/"
    utils.save_images_to_folder(SAVE_FOLDER,"not_part_list_1",not_part_list)
    
    #utils.view_sample_images_from_list(part_list,10)
    #utils.view_sample_images_from_list(not_part_list,10)

    
if __name__ == "__main__":
    sys.exit(main())