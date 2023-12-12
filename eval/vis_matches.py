from glob import glob
import numpy as np
import os
import cv2

if __name__ == "__main__":
    EVAL = "/home/cadar/Documents/Github/DALF_Simulator/eval/nonrigid_eval/"
    eval_folders = sorted(glob(EVAL + "gt_*/"))

    match_images_dict = {}
    names = []
    for folder in eval_folders:
        model_name = os.path.basename(os.path.normpath(folder))
        names.append(model_name)
        for dataset in sorted(glob(os.path.join(folder + '/*/'))):
            dataset_name = os.path.basename(os.path.normpath(dataset))
            match_images = sorted(glob(os.path.join(dataset, '*/*_match.png')))
            for match_image in match_images:
                parent_folder = os.path.basename(os.path.dirname(match_image))
                match_id = os.path.basename(os.path.normpath(match_image))
                key = f"{dataset_name}/{parent_folder}_{match_id}"


                if key not in match_images_dict:
                    match_images_dict[key] = []

                match_images_dict[key].append(match_image)

    for key in match_images_dict:
        images = [cv2.imread(image) for image in match_images_dict[key]]
        # add name to each image
        for i, image in enumerate(images):
            cv2.putText(image, names[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        for i, image in enumerate(images):
            images[i] = cv2.resize(image, None, fx=0.8, fy=0.8)

        images = np.concatenate(images, axis=0)
        cv2.imshow("Matches", images)
        cv2.waitKey(0)
