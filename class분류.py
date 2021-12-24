import os
is_aug = False

bb_directory = "bbox"
land_directory = "land"
masked = "masked"
img_path = "C:/Users/jinyeol/Desktop/Training_backup"
# land_mark = "C:/Users/ddcfd/PycharmProjects/face_detection/face_sdk/api_usage/temp/test1_landmark_res.txt"
# buf = open(land_mark, "r")
# line = buf.readline().strip().split()
train_class= "C:/Users/jinyeol/PycharmProjects/face_detection/face_sdk/api_usage/temp/ohmygirl_train.txt"
class_num = -1

for root, dirs, files in os.walk(img_path):
    pass_dir = root.split("\\")
    if pass_dir[-1] == bb_directory or pass_dir[-1] == land_directory or pass_dir[-1] == masked:
        continue
    # for dir in dirs:
    #
    #     if dir == bb_directory or dir == land_directory or dir == masked:
    #         continue
    #     if not os.path.exists(root + "/" + dir + "/" + masked):
    #         os.makedirs(root + "/" + dir + "/" + masked)

    for file in files:
        if len(file) > 0:
            with open(train_class,"a") as f:
                f.write(root+"/"+file+" "+str(class_num)+"\n")
            # face_lms = [float(num) for num in line]
            # face_masker = FaceMasker(is_aug)
            # print(root + "/" + masked + "masked_" + file)
            # face_masker.add_mask_one(root + "/" + file, face_lms, template_name, root + "/" + masked + "masked_" + file)

            # line = buf.readline().strip().split()
    class_num += 1
