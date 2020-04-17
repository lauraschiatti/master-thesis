BOOST_DIR = /usr/local

SRC_DIR = ../../src

CXX = g++
CFLAGS = -O3 -g -std=c++11 #-shared -fPIC
LDFLAGS= -lpthread -lboost_serialization-mt -lboost_iostreams-mt -lglog -lgflags 
INCLUDE = -I$(SRC_DIR) -I$(BOOST_DIR)/include 
LIBS = -L$(BOOST_DIR)/lib -Wl,-rpath $(BOOST_DIR)/lib 

BIN =  main 
SOURCES = main.cpp
OBJ = $(SOURCES:.cpp=.o)

all:  $(BIN) 

$(BIN) : $(OBJ)
	$(CXX) $(CFLAGS) $(INCLUDE) $(OBJ) $(LIBS) $(LDFLAGS) -o $@ 

.cpp.o: 
	# $(CXX) $(INCLUDE) -c $(CFLAGS) $< -o $@ 
	$(CXX) $(INCLUDE) -c $(CFLAGS) $^ -o $@

clean:
	$(RM) $(BIN) $(OBJ) 
	$(RM) $(SRC_DIR)/apps/log/*.log*

test:
	make clean && make && ./main