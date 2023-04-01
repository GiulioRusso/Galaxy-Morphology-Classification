import csv

import cv2
import os

dataset_augmented = os.path.join("augmented")
annotation_augmented_path = os.path.join("annotations", "all-10-augmented.csv")

index = 18733  # starting point of the new images index
class_to_augment = 0
start = 0  # start index of specified class
end = 500  # end index of specified class
number_of_augmented_images = 400  # how many images we want to add?
final_index = index + number_of_augmented_images  # upper bound index

for i in range(start, end+1):

    if index < final_index:

        image_name = "Galaxy10_DECals-dataset-" + str(i).zfill(5) + ".png"

        # read the input image
        image = cv2.imread(os.path.join("images", image_name))

        # 7 possible rotation (new samples) for each image
        rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_90_flipped_h = cv2.flip(rotated_90, 1)
        rotated_90_flipped_v = cv2.flip(rotated_90, 0)
        rotated_180 = cv2.rotate(rotated_90, cv2.ROTATE_90_CLOCKWISE)
        rotated_180_flipped_h = cv2.flip(rotated_180, 1)
        rotated_180_flipped_v = cv2.flip(rotated_180, 0)
        rotated_270 = cv2.rotate(rotated_180, cv2.ROTATE_90_CLOCKWISE)
        # rotated_270_flipped_h = cv2.flip(rotated_270, 1) # same of rotated_90_flipped_v
        # rotated_270_flipped_v = cv2.flip(rotated_270, 0) # same of rotated_90_flipped_h

        with open(annotation_augmented_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # check the file, because the first row written is added in append to the last row already written in the
            # .csv, so it has to be checked and corrected after the launch

            # for each imwrite, index increase to append the new image into the annotations file
            # after each imwrite the annotation file is updated with FILENAME | CLASS | SPLIT
            # since we have 7 new samples, the split add 4 in train, 1 in validation and 2 in test

            # 1
            index += 1
            image_augmented_name = "Galaxy10_DECals-dataset-" + str(index).zfill(5)
            cv2.imwrite(os.path.join(dataset_augmented, image_augmented_name + ".png"), rotated_90)
            writer.writerow([image_augmented_name, class_to_augment, 'train'])

            # 2
            index += 1
            image_augmented_name = "Galaxy10_DECals-dataset-" + str(index).zfill(5)
            cv2.imwrite(os.path.join(dataset_augmented, image_augmented_name + ".png"), rotated_90_flipped_h)
            writer.writerow([image_augmented_name, class_to_augment, 'train'])

            # 3
            index += 1
            image_augmented_name = "Galaxy10_DECals-dataset-" + str(index).zfill(5)
            cv2.imwrite(os.path.join(dataset_augmented, image_augmented_name + ".png"), rotated_90_flipped_v)
            writer.writerow([image_augmented_name, class_to_augment, 'train'])

            # 4
            index += 1
            image_augmented_name = "Galaxy10_DECals-dataset-" + str(index).zfill(5)
            cv2.imwrite(os.path.join(dataset_augmented, image_augmented_name + ".png"), rotated_180)
            writer.writerow([image_augmented_name, class_to_augment, 'train'])

            # 5
            index += 1
            image_augmented_name = "Galaxy10_DECals-dataset-" + str(index).zfill(5)
            cv2.imwrite(os.path.join(dataset_augmented, image_augmented_name + ".png"), rotated_180_flipped_h)
            writer.writerow([image_augmented_name, class_to_augment, 'validation'])

            # 6
            index += 1
            image_augmented_name = "Galaxy10_DECals-dataset-" + str(index).zfill(5)
            cv2.imwrite(os.path.join(dataset_augmented, image_augmented_name + ".png"), rotated_180_flipped_v)
            writer.writerow([image_augmented_name, class_to_augment, 'test'])

            # 7
            index += 1
            image_augmented_name = "Galaxy10_DECals-dataset-" + str(index).zfill(5)
            cv2.imwrite(os.path.join(dataset_augmented, image_augmented_name + ".png"), rotated_270)
            writer.writerow([image_augmented_name, class_to_augment, 'test'])

            # cv2.imwrite(os.path.join(dataset_augmented, "rotated_270_flipped_h.png"), rotated_270_flipped_h) # no need
            # cv2.imwrite(os.path.join(dataset_augmented, "rotated_270_flipped_v.png"), rotated_270_flipped_v) # no need

# images added:
# class 4: + 1001 images from 17733 to 18733
# class 0: + 406 images from 18734 to 19139