import cv2
import os
import numpy as np

def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))

os.makedirs('landmarks_t', exist_ok=True)
# os.makedirs('original', exist_ok=True)
datas = np.load("check.npy")
count=0
for data in datas:
    black_image = np.zeros((256,256,3), np.uint8)
    landmarks = [[data[0][0], data[1][0]]] 
    for i in range(1,68):
        landmarks.append([data[0][i], data[1][i]])
    jaw = reshape_for_polyline(landmarks[0:17])
    left_eyebrow = reshape_for_polyline(landmarks[22:27])
    right_eyebrow = reshape_for_polyline(landmarks[17:22])
    nose_bridge = reshape_for_polyline(landmarks[27:31])
    lower_nose = reshape_for_polyline(landmarks[30:35])
    left_eye = reshape_for_polyline(landmarks[42:48])
    right_eye = reshape_for_polyline(landmarks[36:42])
    outer_lip = reshape_for_polyline(landmarks[48:60])
    inner_lip = reshape_for_polyline(landmarks[60:68])

    color = (255, 255, 255)
    thickness = 3

    cv2.polylines(black_image, [jaw], False, color, thickness)
    cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [nose_bridge], False, color, thickness)
    cv2.polylines(black_image, [lower_nose], True, color, thickness)
    cv2.polylines(black_image, [left_eye], True, color, thickness)
    cv2.polylines(black_image, [right_eye], True, color, thickness)
    cv2.polylines(black_image, [outer_lip], True, color, thickness)
    cv2.polylines(black_image, [inner_lip], True, color, thickness)

    ## Display the resulting frame
    print(count)
    cv2.imwrite("tmp/encoder_weights/{}.png".format(count), black_image)
    
    count += 1

# path='/Users/adityamehndiratta/pix2pix-tensorflow/robert_test/images/'
# dest='/Users/adityamehndiratta/pix2pix-tensorflow/robert_target/'
# # i=0
# for filename in os.listdir(path):
    
#     temp=filename[5:]
#     temp=temp[:-4]
#     x=(int(temp))
#     my_source =path + filename
#     my_dest = str(x) + ".jpg"
#     print(my_dest)
#     # rename() function will
#     # rename all the files
#     os.rename(my_source, my_dest)
#     # i=i+1

# for filename in os.listdir(path):
#     temp=filename[-11]
#     if(temp=='o'):
#         my_source = path + filename
#         img=cv2.imread(my_source)
#         temp=dest+filename
#         cv2.imwrite(temp,img)
#     # rename() function will
#     # rename all the files
#     # i=i+1

cv2.destroyAllWindows()

