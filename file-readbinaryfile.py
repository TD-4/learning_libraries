

filename = "E:\\点灯数据\\Midle\\lmdb\\img_train_lmdb\\mean.binaryproto"
file = open(filename, "br")

for f in file.readlines():

    print(f.decode(encoding='utfxxx8'))