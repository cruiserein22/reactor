import cv2
import numpy as np
from PIL import Image, ImageDraw

from torchvision.transforms.functional import to_pil_image

from scripts.reactor_logger import logger
from scripts.reactor_inferencers.bisenet_mask_generator import BiSeNetMaskGenerator
from scripts.reactor_entities.face import FaceArea
from scripts.reactor_entities.rect import Rect
from insightface.app.common import Face

from scripts.reactor_inferencers.mask_generator import MaskGenerator

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (0, 128, 128),
]

def color_generator(colors):
    while True:
        for color in colors:
            yield color


# def process_face_image(
#         face: FaceArea,
#         exclude_mouth: bool = False,  # New parameter to control mouth exclusion
#         **kwargs,
#     ) -> Image:
#         image = np.array(face.image)
#         overlay = image.copy()
#         color_iter = color_generator(colors)
        
#         # Draw a rectangle over the entire face
#         cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), next(color_iter), -1)
#         l, t, r, b = face.face_area_on_image
#         cv2.rectangle(overlay, (l, t), (r, b), (0, 0, 0), 10)
        
#         print("checking landsmarks_on_image:",face.landmarks_on_image)
#         if face.landmarks_on_image is not None:
#             for landmark in face.landmarks_on_image:
#                 # Check if the landmark is part of the mouth, if exclude_mouth is True
#                 if exclude_mouth and is_mouth_landmark(landmark):
#                     continue  # Skip mouth landmarks
                
#                 # Draw a circle for each landmark
#                 cv2.circle(overlay, (int(landmark.x), int(landmark.y)), 6, (0, 0, 0), 10)
        
#         alpha = 0.3
#         output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
#         return Image.fromarray(output)


def process_face_image(
        face: FaceArea,
        exclude_bottom_half: bool = False,  # New parameter to control exclusion of bottom half
        **kwargs,
    ) -> Image:
        image = np.array(face.image)
        overlay = image.copy()
        color_iter = color_generator(colors)
        
        # Draw a rectangle over the entire face
        cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), next(color_iter), -1)
        l, t, r, b = face.face_area_on_image
        cv2.rectangle(overlay, (l, t), (r, b), (0, 0, 0), 10)
        
        print("checking landmarks_on_image:", face.landmarks_on_image)
        if face.landmarks_on_image is not None:
            # Determine the y-coordinate of the nose to define the exclusion boundary
            nose_y = get_nose_y_coordinate(face.landmarks_on_image)

            for landmark in face.landmarks_on_image:
                # Exclude everything below the nose if exclude_bottom_half is True
                if exclude_bottom_half and int(landmark.y) >= nose_y:
                    continue  # Skip landmarks in the bottom half
                
                # Draw a circle for each landmark
                cv2.circle(overlay, (int(landmark.x), int(landmark.y)), 6, (0, 0, 0), 10)
        
        alpha = 0.3
        output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        return Image.fromarray(output)

def get_nose_y_coordinate(landmarks):
    """
    Determine the y-coordinate of the nose landmark to define the boundary for exclusion.
    This assumes that landmarks are provided and one of them represents the nose.
    Adjust this function based on how your landmarks are structured and named.
    """
    nose_y = None
    for landmark in landmarks:
        if is_nose_landmark(landmark):  # Implement this function based on your landmarks
            nose_y = int(landmark.y)
            break
    return nose_y if nose_y is not None else 0  # Default to 0 if nose isn't found

def is_nose_landmark(landmark):
    """
    Determine if a given landmark represents the nose.
    You'll need to adjust the logic here based on your specific landmark detection system.
    """
    # Placeholder condition; replace with your actual condition for identifying a nose landmark.
    print("landmark",landmark)
    return landmark.part == "Nose"  # Adjust based on your system

def is_mouth_landmark(landmark):
    # This function needs to be tailored to your specific landmark system
    # Typically, you'd check if the landmark's index or name indicates it's part of the mouth
    # For example:
    # return landmark.part in ["Mouth_Lower_Lip", "Mouth_Upper_Lip"]  # Adjust based on your system
    print("landmark",landmark)
    return False  # Placeholder; implement your own logic here


def apply_face_mask(swapped_image:np.ndarray,target_image:np.ndarray,target_face,entire_mask_image:np.array)->np.ndarray:
    logger.status("Correcting Face Mask")
    mask_generator =  BiSeNetMaskGenerator()
    face = FaceArea(target_image,Rect.from_ndarray(np.array(target_face.bbox)),1.6,512,"")
    face_image = np.array(face.image)
    process_face_image(face)
    face_area_on_image = face.face_area_on_image
    mask = mask_generator.generate_mask(
        face_image,
        face_area_on_image=face_area_on_image,
        affected_areas=["Face"],
        mask_size=0,
        use_minimal_area=True
    )
    mask = cv2.blur(mask, (12, 12))
    # """entire_mask_image = np.zeros_like(target_image)"""
    larger_mask = cv2.resize(mask, dsize=(face.width, face.height))
    entire_mask_image[
        face.top : face.bottom,
        face.left : face.right,
    ] = larger_mask
   
    result = Image.composite(Image.fromarray(swapped_image),Image.fromarray(target_image), Image.fromarray(entire_mask_image).convert("L"))
    return np.array(result)


def rotate_array(image: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def rotate_image(image: Image, angle: float) -> Image:
    if angle == 0:
        return image
    return Image.fromarray(rotate_array(np.array(image), angle))


def correct_face_tilt(angle: float) -> bool:
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle > 40


def _dilate(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.dilate(arr, kernel, iterations=1)


def _erode(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.erode(arr, kernel, iterations=1)


def dilate_erode(img: Image.Image, value: int) -> Image.Image:
    """
    The dilate_erode function takes an image and a value.
    If the value is positive, it dilates the image by that amount.
    If the value is negative, it erodes the image by that amount.

    Parameters
    ----------
        img: PIL.Image.Image
            the image to be processed
        value: int
            kernel size of dilation or erosion

    Returns
    -------
        PIL.Image.Image
            The image that has been dilated or eroded
    """
    if value == 0:
        return img

    arr = np.array(img)
    arr = _dilate(arr, value) if value > 0 else _erode(arr, -value)

    return Image.fromarray(arr)

def mask_to_pil(masks, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (width, height) of the original image
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]

def create_mask_from_bbox(
    bboxes: list[list[float]], shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def mask_bottom_half_of_face(mask: np.ndarray, face_area: FaceArea) -> np.ndarray:
    """
    Modify the mask to cover only the bottom half of the face from the nose down.

    Parameters:
    - mask (np.ndarray): The original face mask.
    - face_area (FaceArea): The FaceArea object containing the Rect with landmarks.

    Returns:
    - np.ndarray: The modified mask with only the bottom half of the face covered.
    """
    rect = face_area.face_area  # Extract the Rect from the FaceArea
    if rect.landmarks and rect.landmarks.nose:
        # Use the nose landmark to define the starting point of the bottom half
        nose = rect.landmarks.nose

        # Calculate the starting y-coordinate for the mask
        y_start = int(nose.y * face_area.scale_factor)

        # Calculate the height of the face to find the bottom
        face_height = int(face_area.height * face_area.scale_factor)

        # Set the top half of the face to 0 (unmasked) in the mask
        mask[:y_start, :] = 0

        # Optionally, if you want to ensure the mask only covers the area up to the chin:
        # Determine the chin's y-coordinate (you might need a chin landmark or another method)
        # y_chin = int(chin.y * face_area.scale_factor)
        # mask[y_chin:, :] = 0  # Uncomment and adjust if you have a chin landmark or method

    return mask

def exclude_mouth_from_mask(mask: np.ndarray, face_area: FaceArea) -> np.ndarray:
    """
    Modify the mask to exclude the mouth region based on the provided landmarks in FaceArea.

    Parameters:
    - mask (np.ndarray): The original face mask.
    - face_area (FaceArea): The FaceArea object containing the Rect with landmarks.

    Returns:
    - np.ndarray: The modified mask with the mouth area excluded.
    """
    rect = face_area.face_area  # Extract the Rect from the FaceArea
    if rect.landmarks and rect.landmarks.mouth1 and rect.landmarks.mouth2:
        # Use the mouth landmarks to define the exclusion area
        mouth1, mouth2 = rect.landmarks.mouth1, rect.landmarks.mouth2

        # Calculate the bounding box for the mouth
        x_min, y_min = min(mouth1.x, mouth2.x), min(mouth1.y, mouth2.y)
        x_max, y_max = max(mouth1.x, mouth2.x), max(mouth1.y, mouth2.y)

        # Adjust for the scale and position of the face in the entire image
        x_min, y_min, x_max, y_max = [int(val * face_area.scale_factor) for val in [x_min, y_min, x_max, y_max]]

        # Set the mouth region to 0 (black) in the mask
        mask[y_min:y_max, x_min:x_max] = 0

    return mask




def apply_face_mask_with_exclusion(swapped_image: np.ndarray, target_image: np.ndarray, target_face: Face, entire_mask_image: np.ndarray) -> np.ndarray:
    """
    Apply the face mask with an exclusion zone for the mouth.

    Parameters:
    - swapped_image (np.ndarray): The image with the swapped face.
    - target_image (np.ndarray): The target image where the face will be placed.
    - target_face: Face with bbox The bounding box of the target face.
    - entire_mask_image (np.ndarray): The initial entire mask image.

    Returns:
    - np.ndarray: The result image with the face swapped, excluding the mouth region.
    """

    logger.status("mask_bottom_half_of_face Mouth Mask")


    # Extract the bbox array from the target_face object
    target_face_bbox = target_face.bbox if hasattr(target_face, 'bbox') else None
    
    if target_face_bbox is None:
        logger.error("No bounding box found in the target face object.")
        return target_image  # or handle this scenario appropriately

    # Now you can safely create a Rect object from the bbox array
    rect = Rect.from_ndarray(np.array(target_face_bbox))
 
    face = FaceArea(target_image, rect, 1.6, 512, "")
    face_image = np.array(face.image)
    process_face_image(face,exclude_bottom_half=True)
    face_area_on_image = face.face_area_on_image
# Then call the generate_mask method with all required arguments
    mask_generator =  BiSeNetMaskGenerator()
    mask = mask_generator.generate_mask(
        face_image=face_image,  # make sure this is the first required positional argument
        face_area_on_image=face_area_on_image,
        affected_areas=["Face"],
        mask_size=0,
        use_minimal_area=True
    )
    mask = cv2.blur(mask, (12, 12))

    # Modify the mask to exclude the mouth using the FaceArea object.
    # mask = exclude_mouth_from_mask(mask, face)
    mask = mask_bottom_half_of_face(mask, face)
    
    larger_mask = cv2.resize(mask, dsize=(face.width, face.height))
    entire_mask_image[
        face.top:face.bottom,
        face.left:face.right,
    ] = larger_mask

    result = Image.composite(Image.fromarray(swapped_image), Image.fromarray(target_image), Image.fromarray(entire_mask_image).convert("L"))
    return np.array(result)





# Use apply_face_mask_with_exclusion instead of apply_face_mask when performing the face swap