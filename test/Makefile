# GTEST_DIR = ../../gtest-1.7.0
# osx
GTEST_DIR = ~/googletest
# ubuntu
# GTEST_DIR = /usr 
BOOST_DIR = /usr/local
GLOG_GFLAGS_DIR = /usr/local

SRC_DIR = ../src

# compiler
CXX = g++  # C++
# CC = $(CXX)

# compiler flags
CFLAGS = -O3 -g -std=c++11 #-shared -fPIC

# linker flags
LDFLAGS= -lpthread -lboost_serialization -lboost_iostreams -lglog

# If we're on OS X, make sure that globals aren't stripped out.
# HOST_ARCH = $(shell uname -m) # get host architecture

# ifeq ($(HOST_ARCH),X86_64)
# 	$(info $$HOST_ARCH is [${HOST_ARCH}])
# 	LDFLAGS += -all_load
# 	CFLAGS += -framework CoreFoundation
# endif

INCLUDE = -I$(SRC_DIR) -I$(BOOST_DIR)/include -I$(GTEST_DIR)/include -I$(GLOG_GFLAGS_DIR)/include

# list of libraries to link into executable; -l tells the linker which
# library to link into the executable
LIBS = $(GTEST_DIR)/lib/libgtest.a -L$(GLOG_GFLAGS_DIR)/lib -L$(BOOST_DIR)/lib

# project name
BIN = run_all_tests

# source file
SOURCES = main_test.cpp

# OBJECTS is assigned the value of SOURCES, after substituting
# every .cpp at the end of an item with .o.
OBJ = $(SOURCES:.cpp=.o)

# all is the default rule
all: $(SOURCES) $(BIN)

$(BIN) : $(OBJ)
	$(CXX) $(CFLAGS) $(INCLUDE) $(OBJ) -o $@ $(LIBS) $(LDFLAGS)

# compile every .c file into .o
.cpp.o:
	# $(CXX) $(INCLUDE) -c $(CFLAGS) $< -o $@

	# use $^ to link all the OBJ files
	$(CXX) $(INCLUDE) -c $(CFLAGS) $^ -o $@

# remove all obj files
clean:
	$(RM) $(BIN)* $(OBJ) *~ *.dSYM test_data/*.bin

test:
	make clean && make && ./run_all_tests