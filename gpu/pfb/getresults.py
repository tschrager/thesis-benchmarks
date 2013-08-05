import subprocess

iterations = "100x"
gpu = "gtx580"
#gpu = "test"

subprocess.call(["make", "all"])
taps="4"
print "Testing pfb 4 taps..."
r2cfile = open('results/pfb_'+taps+'_'+gpu+'_'+iterations,'w')
p = subprocess.Popen(["./pfbtest"],stdout=r2cfile,stderr=subprocess.STDOUT)
p.wait()
print "Done with pfb 4 taps"

