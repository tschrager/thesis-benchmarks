import subprocess

iterations = "100x"
gpu = "gtx580"
#gpu = "test"

subprocess.call(["make", "all"])

print "Testing r2c..."
r2cfile = open('results/r2c_'+gpu+'_'+iterations,'w')
p = subprocess.Popen(["./cuffttestr2c"],stdout=r2cfile,stderr=subprocess.STDOUT)
p.wait()
print "Done with r2c"

print "Testing c2c..."
c2cfile = open('results/c2c_'+gpu+'_'+iterations,'w')
p = subprocess.Popen(["./cuffttestc2c"],stdout=c2cfile,stderr=subprocess.STDOUT)
p.wait()
print "Done with c2c"
