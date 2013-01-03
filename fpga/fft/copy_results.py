import glob
import shutil

for file in glob.iglob('*/sysgen/xflow/*_map.map'):
    shutil.copy(file,'results')
for file in glob.iglob('*/sysgen/xflow/*.twr'):
    shutil.copy(file,'results')

