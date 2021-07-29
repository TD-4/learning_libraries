import os
import shutil
import sys
import time
import datetime
from datetime import timedelta


def delete_raw(path=""):
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for dir_ in dirs:
                if dir_ == "Raw" or dir_ == "raw":
                    shutil.rmtree(os.path.join(root, dir_))
        print("delete success!!!")
    except BaseException:
        print("Error!!! ， 请联系开发人员")
    finally:
        # os.system("pause")
        pass


def delete_raw_data(path="", grid="30", raw="30"):
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for dir_ in dirs:
                if dir_ == "Raw":
                    folder_date = root.split("\\")[-1][:8]
                    today = time.strftime('%Y%m%d', time.localtime(time.time()))
                    folder_date = datetime.datetime.strptime(folder_date, "%Y%m%d")  # 字符串转化为date形式
                    today = datetime.datetime.strptime(today, "%Y%m%d")  # 字符串转化为date形式
                    flag_raw = today - timedelta(days=int(raw)) > folder_date
                    if flag_raw:
                        shutil.rmtree(os.path.join(root, dir_))
                if dir_ == "Grid":
                    folder_date = root.split("\\")[-1][:8]
                    today = time.strftime('%Y%m%d', time.localtime(time.time()))
                    folder_date = datetime.datetime.strptime(folder_date, "%Y%m%d")  # 字符串转化为date形式
                    today = datetime.datetime.strptime(today, "%Y%m%d")  # 字符串转化为date形式
                    flag_grid = today - timedelta(days=int(grid)) > folder_date
                    if flag_grid:
                        shutil.rmtree(os.path.join(root, dir_))
        print("delete success!!!")
    except BaseException as ex:
        print("Error!!! ， 请联系开发人员")
        print(ex)
    finally:
        # os.system("pause")
        pass


if __name__ == "__main__":
    # delete_raw(path=r"D:\COG\LCDSystemTS")
    raw = sys.argv[1]
    grid = sys.argv[2]
    delete_raw_data(path=r"D:\COG\LCDSystemTS", grid=str(grid), raw=str(raw))
