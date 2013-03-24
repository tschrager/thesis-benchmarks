CC = nvcc
NVIDIA_SDK_DIR = /opt/local/cuda-5.0/samples
CUTIL_LIB_NAME = cutil_x86_64
CFLAGS = -g -O2
#DEBUG = -D DEBUG
CUDA_INCLUDE_HEADER = -I/usr/local/cuda/include -I$(NVIDIA_SDK_DIR)/common/inc 
CUDA_INCLUDE_LIB = -L/usr/local/cuda/lib64 -L$(NVIDIA_SDK_DIR)/commonlib/linux/x86_64 -lcufft -lcuda 

TARGETS = pfbtest
.PHONY: clean all

default: pfbtest

all: $(TARGETS)

%.o: %.cu Makefile
	$(CC) $(CFLAGS) $(CUDA_INCLUDE_HEADER) $(DEBUG) -c $<
	
%.o: %.c Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<

$(TARGETS): %: %.o
	$(CC) $(CFLAGS) $(CUDA_INCLUDE_LIB) $(DEBUG) $*.o -o $*

clean:
	@rm -vf *.o
	
realclean: clean
	@rm -vf $(LIB_TARGETS)
	@rm -vf $(TARGETS)
