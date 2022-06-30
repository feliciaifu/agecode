import os
import pprint

import imageio
class people:
    def __init__(self):
        self.name=""
        self.images=[[]]
if __name__ == "__main__":
    # ---------------------#
    #   训练集所在的路径
    # ---------------------#
    datasets_path = r"/home/yifu/下载/images"
    file_list = open('data_path.txt', 'w')
    files=os.listdir(datasets_path)
    sorted(files)
    for i in range(0, len(files)):
        #TODO
        people_name=files[i][:-9]
        out_dir="/home/yifu/下载/images-out/"+people_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        filename="/"+str(i)+".png"
        image=imageio.imread(datasets_path+"/"+files[i])
        str3=out_dir+filename
        pprint.pprint(str3)
        imageio.imsave(str3,image)
