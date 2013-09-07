import re
import os
import numpy
from matplotlib import pyplot as plt

ants = [8,16,32,64,128]
dsps = [80,144,272,528,1040]

ants_interp = numpy.linspace(1,140)
dsps_interp = 8*ants_interp+16
plt.bar(ants,dsps)
plt.plot(ants_interp,dsps_interp, 'g--')
plt.text(75,800,'DSPs = 8*Antennas+16',rotation=0)
plt.xlabel('Antennas')
plt.ylabel('DSPs')
#plt.plot(ants,dsps, 'rx')
#plt.show()
plt.savefig('xeng_fpga_bench.png')   

