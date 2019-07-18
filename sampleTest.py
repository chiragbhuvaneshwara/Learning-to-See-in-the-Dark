import os
cwd = os.getcwd()
l = os.readlink(cwd+'/dataset')
l = os.listdir(l)
#l = os.listdir(cwd+'/dataset')
print(cwd)
print(l)
