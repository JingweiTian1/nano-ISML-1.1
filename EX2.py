import os
import shutil

dirlist = os.listdir("all_data")
for name in dirlist:
    li2 = os.listdir("all_data//"+name)
    for ls in li2:
        print(ls)
        if ls[len(ls)-9:len(ls)]=="c2-c3.png":
            shutil.copy("all_data//"+name+"//"+ls,"dataset//last//"+name+"_"+ls[0:5]+ls[6:]+".png")
            print("yuan_all//"+name+"_"+ls[0:5]+ls[6:]+".png")

