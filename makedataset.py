import os
import shutil
import re

current_path = "/home/panhan/oxbuild"
print('当前目录：'+current_path)

filename_list = os.listdir(current_path)
print('当前目录下文件：',filename_list)
print('正在分类整理进文件夹ing...')
for filename in filename_list:
    try:
        name1, name2 = filename.split('.')
        if name2 == 'jpg' or name2 == 'png':
            sname = name1.rsplit("_",1)
            print(sname)
            wname=sname[0]
            wdirname=os.path.join(current_path, wname)
            if not os.path.exists(wdirname):
                os.makedirs(wdirname)
            shutil.move(os.path.join(current_path, filename), wdirname)
            # try:
            #     # sname=name1.split('_')
            #     print(sname[:-1])
            #     os.mkdir(sname[:-1])
            #     print('创建文件夹'+sname[:-1])
            # except:
            #     pass
            # try:
            #     shutil.move(current_path+'\\'+filename,current_path+'\\'+sname[1])
            #     print(filename+'转移成功！')
            # except Exception as e:
            #     print('移动失败:' + e)
    except:
        pass

print('整理完毕！')


