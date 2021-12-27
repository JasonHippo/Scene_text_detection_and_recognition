
badword = ["!","#","$","%","&","'","(",")","*","+",",","-",".","/",":",";","<","=",">","?","@","[","/","]","^","_","`","{","|","}","~","の","®"]


ch = []
with open('/nfs/Workspace/deep-text-recognition-benchmark/data_tbrain_and_AE/train/gt.txt', 'r', encoding="utf-8") as txt:
    lines = txt.readlines()
    for l in lines:
        label = l.split(' ')[-1].rstrip("\n")
        print('label:{}'.format(label))
        for c in label:
            if 'a' <= c <= "z" or 'A' <= c <='Z' or '0' <= c <='9':
                continue #如果是英文或數字就跳過
            if c in ch:
                continue
            if ch in badword:
                continue
            print(c)
            ch.append(c)
            print(c, len(ch))

print(len(ch))
with open('/nfs/Workspace/deep-text-recognition-benchmark/data_tbrain_and_AE/total_ch.txt', 'w', encoding="utf-8") as ch_txt:
    for c in ch:
        ch_txt.write(c)
