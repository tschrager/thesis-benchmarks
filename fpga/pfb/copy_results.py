import glob
import shutil
import re
import os.path


blockname = 'pfb'
technologyindex = 6

#for directory in glob.iglob('')
technames = set([])
for builddirectory in glob.iglob(blockname+'*/'):
    technames.add(re.split('/|_',builddirectory)[6])
    

for techname in list(technames):
#    print techname
    techdir = os.path.dirname('results/'+techname+'/')
    if not os.path.exists(techdir):
        os.makedirs(techdir)

for file in glob.iglob('*/sysgen/xflow/*_map.map'):
    techname = re.split('/|_',file)[technologyindex]
    shutil.copy(file,'results/'+techname+'/')
    
for file in glob.iglob('*/sysgen/xflow/*.twr'):
    techname = re.split('/|_',file)[technologyindex]
    shutil.copy(file,'results/'+techname+'/')

