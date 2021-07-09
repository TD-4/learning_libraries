from imagededup.methods import PHash, CNN, DHash
import json, os

phasher = DHash()

duplicates = phasher.find_duplicates_to_remove(image_dir=os.path.join("F:\\Data\\lcd\\edge\\20210627", "12_DBLM"),
                                                       max_distance_threshold=4,
                                                       outfile='my_duplicates_.json')

# path = "/root/old_slim"
#
# folders = os.listdir(path=path)

# # 重命名所有文件
# for folder in folders:
#     all_images = os.listdir(os.path.join(path, folder))
#     if all_images == []:
#         continue
#
#     for i, image in enumerate(all_images):
#         # if os.path.isdir(os.path.join(path, folder, image)):
#         #     os.remove(os.path.join(path, folder, image))
#         #     continue
#         os.rename(os.path.join(path, folder, image), os.path.join(path, folder, str(i)+".bmp"))

# 获得所有要删除的文件名
# for folder in folders:
#     all_images = os.listdir(os.path.join(path, folder))
#     if all_images == []:
#         continue
#     duplicates = phasher.find_duplicates_to_remove(image_dir=os.path.join(path, folder),
#                                                max_distance_threshold=8,
#                                                outfile='my_duplicates_{}.json'.format(folder))


# for folder in folders:
#     all_images = os.listdir(os.path.join(path, folder))
#     if all_images == []:
#         continue
#     with open("./my_duplicates_{}.json".format(folder),'r') as load_f:
#          load_dict = json.load(load_f)
#     for name in load_dict:
#         os.remove(os.path.join(path ,folder, name))