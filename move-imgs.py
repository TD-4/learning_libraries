import os
import shutil


def move_bmp(json_path="", bmp_path=""):
    all_jsons = os.listdir(json_path)
    for json_ in all_jsons:
        shutil.copy(os.path.join(bmp_path, json_[:-4] + "bmp"),
                    os.path.join(json_path, json_[:-4] + "bmp"))


if __name__ == "__main__":
    json_path = r"F:\Data\LandingAI\5--付发\白斑-造-Err-第二批437"
    bmp_path = r"F:\Data\LandingAI\5--付发\白斑-造-Err-第二批452"
    move_bmp(json_path, bmp_path)