import os
path = 'C:/Users/Saurabh/Desktop/26/test'
files = os.listdir(path)
print(files)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, 'test.' + str(i)+'.jpg'))
    i = i+1
