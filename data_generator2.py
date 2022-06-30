
from PIL import Image

import os

abpos=0
dir_name = "/home/yifu/下载/images-out/"
dir_out = "/home/yifu/下载/images-out-out/"
if os.path.isdir(dir_name):
    relpaths=os.listdir(dir_name)
    sorted(relpaths)
    for relpath in relpaths:
        abspath = os.path.join(dir_name, relpath)
        abpos = abpos + 1
        photopos = 0
        if not os.path.exists(dir_out + "/" + str(abpos) + "/"):
            os.mkdir(dir_out+"/"+str(abpos)+"/")
        for relpath1 in os.listdir(abspath):
            abspath1 = os.path.join(abspath, relpath1)
            if os.path.isfile(abspath1) and any(
                    [abspath1.endswith('.' + ty) for ty in ('jpg', 'png', 'JPG', 'PNG', 'jpeg')]):
                img = Image.open(abspath1).convert("RGB")
                photopos=photopos+1
                outfile=dir_out+"/"+str(abpos)+"/"+str(photopos)+".jpg"
                img.save(outfile)