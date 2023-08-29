# this Makefile is configured by the 'config.inc' that describes your system env.

ifndef CONFIG_FILE
CONFIG_FILE = config.inc
endif

ifeq (,$(wildcard $(CONFIG_FILE)))
$(info config file $(CONFIG_FILE) does not exist.)
$(error Create $(CONFIG_FILE) from example or specify another config via: make <target> CONFIG_FILE=<config_filename> )          
endif

-include $(CONFIG_FILE)

G++ = $(GCC_ROOT_PATH)/g++
G++FLAGS = -g -std=$(CPPSTD)
ICUDA = -I$(CUDA_ROOT_PATH)/include
IPROJECT = -I source/
IBOOST = -I$(BOOST_ROOT_PATH)/include

LBOOST = -L$(BOOST_ROOT_PATH)/lib
LIBBOOST = -lboost_serialization
LLAPACK = -L$(OPENBLAS_ROOT_PATH)/lib -lopenblas

deb:
	$(G++) $(G++FLAGS) $(IPROJECT) $(IBOOST) source/test_inst/test_multi_precision/test_multi_precision.cpp -o test_multi_precision.bin 
