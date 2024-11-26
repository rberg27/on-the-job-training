from PIL import Image
import copy
from typing import Callable
import utils
import random
    
def augment(init_part_list:[Image],
            init_not_part_list:[Image],
            part_augment_method:Callable[[Image],Image],
            not_part_augment_method:Callable[[Image],Image],
            part_list_size:int,
            full_not_part_ratio:float=1.0) -> (list[Image], list[Image]):
    
    full_part_list = augment_list(init_part_list,part_augment_method,part_list_size)
    not_part_list_size = utils.get_int_from_ratio(full_not_part_ratio,part_list_size)
    full_not_part_list = augment_list(init_not_part_list,not_part_augment_method,not_part_list_size)
    
    return full_part_list, full_not_part_list


def augment_list(image_list:list[Image],
                 augment_method:Callable[[Image],Image],
                 size:int) -> list[Image]:
    new_list = copy.deepcopy(image_list)
    while len(new_list) < size:
        image = utils.get_random_image(image_list)
        new_image = augment_method(image)
        new_list.append(new_image)
        
    return new_list

def create_crop_method(image_and_annotations:[(Image,list[dict])],
                       is_part:bool=True):
    
    def crop_augment(_: Image) -> Image:
        random_idx = random.randint(0,len(image_and_annotations)-1)
        random_image, random_annotations = image_and_annotations[random_idx]
        if is_part:
            return utils.generate_part_images(random_image,random_annotations,1)[0]
        else:
            return utils.generate_not_part_images(random_image,random_annotations,1)[0]
        
    return crop_augment


def create_rotate_method():
    
    def rotate_augment(img: Image) -> Image:
        random_rotation = (int)(random.uniform(0,360))
        new_img = rotate_image(img,random_rotation)
        return new_img
        
    return rotate_augment
    
def rotate_image(image:Image,
                 rotation:int):
    new_img = copy.deepcopy(image)
    new_img = new_img.rotate(rotation,expand=True, fillcolor=(255, 255, 255))
    return new_img

def create_flip_method():
    FLIP_DIRECTION = [Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]
    def flip_augment(img: Image) -> Image:
        new_img = copy.deepcopy(img)
        random_direction = random.choice(FLIP_DIRECTION)
        new_img = new_img.transpose(random_direction)
        return new_img
        
    return flip_augment

def main():
    print("augment_data")
    
if __name__ == "__main__":
    sys.exit(main())