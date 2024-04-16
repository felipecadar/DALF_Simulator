
'''
    Minimal snippet to extract DALF features from an image, following
    OpenCV's features2D interface standard.
'''

from modules.models.DALF import DALF_extractor as DALF
import torch
import cv2

network_path='weights/model_ts-fl_final.pth'


dalf = DALF(model=network_path, dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

img1 = cv2.imread('./assets/kanagawa_1.png')
img2 = cv2.imread('./assets/kanagawa_2.png')

kps1, descs1 = dalf.detectAndCompute(img1)
kps2, descs2 = dalf.detectAndCompute(img2)

# ignore keypoints too close to the border of the image
border = 10
kps1_idxs = [i for i, kp in enumerate(kps1) if kp.pt[0] > border and kp.pt[0] < img1.shape[1] - border and kp.pt[1] > border and kp.pt[1] < img1.shape[0] - border]
kps2_idxs = [i for i, kp in enumerate(kps2) if kp.pt[0] > border and kp.pt[0] < img2.shape[1] - border and kp.pt[1] > border and kp.pt[1] < img2.shape[0] - border]

kps1_idxs = kps1_idxs[:10]

kps1 = [kps1[i] for i in kps1_idxs]
kps2 = [kps2[i] for i in kps2_idxs]
descs1 = descs1[kps1_idxs, :]
descs2 = descs2[kps2_idxs, :]

kp_image1 = cv2.drawKeypoints(img1, kps1, None)
kp_image2 = cv2.drawKeypoints(img2, kps2, None)
kp_image = cv2.hconcat([kp_image1, kp_image2])

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.match(descs1, descs2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, flags=2)

cv2.imwrite('matches.png', img3)
cv2.imwrite('kps.png', kp_image)

#cv2.imshow('Matches', img3)
#cv2.imshow('Keypoints', kp_image)
#cv2.waitKey(0)


