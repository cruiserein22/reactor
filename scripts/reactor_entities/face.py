import traceback
import cv2
import numpy as np
from modules import images
from PIL import Image
from scripts.reactor_entities.rect import Point, Rect

class FaceArea:
    def __init__(self, entire_image: np.ndarray, face_area: Rect, face_margin: float, face_size: int, upscaler: str):
        # Initialize the FaceArea with the entire image, a specified face area (Rect object), margin, size, and upscaler.
        self.face_area = face_area
        self.center = face_area.center  # Center point of the face area.
        left, top, right, bottom = face_area.to_square()  # Convert face area to a square for uniformity.

        # Ensure the face area has a margin around it for context.
        self.left, self.top, self.right, self.bottom = self.__ensure_margin(
            left, top, right, bottom, entire_image, face_margin
        )

        # Dimensions of the face area.
        self.width = self.right - self.left
        self.height = self.bottom - self.top

        # Crop and possibly upscale the face image from the entire image.
        self.image = self.__crop_face_image(entire_image, face_size, upscaler)
        self.face_size = face_size  # Desired face size.
        self.scale_factor = face_size / self.width  # Scaling factor for resizing.
        self.face_area_on_image = self.__get_face_area_on_image()  # Actual face area on the cropped image.
        self.landmarks_on_image = self.__get_landmarks_on_image()  # Facial landmarks on the image.

    def __get_face_area_on_image(self):
        # Calculate the face area on the cropped and possibly upscaled image.
        left = int((self.face_area.left - self.left) * self.scale_factor)
        top = int((self.face_area.top - self.top) * self.scale_factor)
        right = int((self.face_area.right - self.left) * self.scale_factor)
        bottom = int((self.face_area.bottom - self.top) * self.scale_factor)
        return self.__clip_values(left, top, right, bottom)

    def __get_landmarks_on_image(self):
        # Adjust the landmarks' positions based on the cropped and possibly upscaled image.
        landmarks = []
        if self.face_area.landmarks is not None:
            for landmark in self.face_area.landmarks:
                landmarks.append(
                    Point(
                        int((landmark.x - self.left) * self.scale_factor),
                        int((landmark.y - self.top) * self.scale_factor),
                    )
                )
        return landmarks

    def __crop_face_image(self, entire_image: np.ndarray, face_size: int, upscaler: str):
        # Crop the face image from the entire image and resize it according to the desired face size.
        cropped = entire_image[self.top : self.bottom, self.left : self.right, :]
        if upscaler:
            return images.resize_image(0, Image.fromarray(cropped), face_size, face_size, upscaler)
        else:
            return Image.fromarray(cv2.resize(cropped, dsize=(face_size, face_size)))

    def __ensure_margin(self, left: int, top: int, right: int, bottom: int, entire_image: np.ndarray, margin: float):
        # Ensure there's a margin around the face area by expanding it proportionally.
        entire_height, entire_width = entire_image.shape[:2]

        side_length = right - left
        margin = min(min(entire_height, entire_width) / side_length, margin)
        diff = int((side_length * margin - side_length) / 2)

        # Adjust the face area with the margin and ensure it doesn't go out of image bounds.
        top = top - diff
        bottom = bottom + diff
        left = left - diff
        right = right + diff

        # Correct positions if they go out of the image boundaries.
        if top < 0:
            bottom = bottom - top
            top = 0
        if left < 0:
            right = right - left
            left = 0

        if bottom > entire_height:
            top = top - (bottom - entire_height)
            bottom = entire_height
        if right > entire_width:
            left = left - (right - entire_width)
            right = entire_width

        return left, top, right, bottom

    def get_angle(self) -> float:
        # Calculate the angle of the face based on the eye positions.
        landmarks = getattr(self.face_area, "landmarks", None)
        if landmarks is None:
            return 0

        eye1 = getattr(landmarks, "eye1", None)
        eye2 = getattr(landmarks, "eye2", None)
        if eye2 is None or eye1 is None:
            return 0

        try:
            # Calculate angle between the eyes.
            dx = eye2.x - eye1.x
            dy = eye2.y - eye1.y
            if dx == 0:
                dx = 1
            angle = np.arctan(dy / dx) * 180 / np.pi

            # Adjust angle based on the quadrant.
            if dx < 0:
                angle = (angle + 180) % 360
            return angle
        except Exception:
            print(traceback.format_exc())
            return 0

    def rotate_face_area_on_image(self, angle: float):
        # Rotate the face area on the image based on the given angle.
        center = [
            (self.face_area_on_image[0] + self.face_area_on_image[2]) / 2,
            (self.face_area_on_image[1] + self.face_area_on_image[3]) / 2,
        ]

        # Define points to represent the face area.
        points = [
            [self.face_area_on_image[0], self.face_area_on_image[1]],
            [self.face_area_on_image[2], self.face_area_on_image[3]],
        ]

        # Calculate rotation matrix and apply it to the points.
        angle = np.radians(angle)
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        points = np.array(points) - center
        points = np.dot(points, rot_matrix.T)
        points += center
        left, top, right, bottom = (int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]))

        # Adjust the face area based on the rotation.
        left, right = (right, left) if left > right else (left, right)
        top, bottom = (bottom, top) if top > bottom else (top, bottom)

        width, height = right - left, bottom - top
        if width < height:
            left, right = left - (height - width) // 2, right + (height - width) // 2
        elif height < width:
            top, bottom = top - (width - height) // 2, bottom + (width - height) // 2
        return self.__clip_values(left, top, right, bottom)

    def __clip_values(self, *args):
        # Ensure that the values don't go beyond the specified face size.
        result = []
        for val in args:
            if val < 0:
                result.append(0)
            elif val > self.face_size:
                result.append(self.face_size)
            else:
                result.append(val)
        return tuple(result)
