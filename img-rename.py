import os


path = "/root/old_slim"

folders = os.listdir(path=path)

for folder in folders:
    all_images = os.listdir(os.path.join(path, folder))
    if all_images == []:
        continue

    for i, image in enumerate(all_images):
        os.rename(os.path.join(path, folder, image), os.path.join(path, folder, str(i)+".bmp"))
        # print("{}->{}".format(os.path.join(path, folder, image), os.path.join(path, folder, str(i)+".bmp")))
        # break
    # break
