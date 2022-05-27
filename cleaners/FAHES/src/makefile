CC = g++ 			#C compiler
CXXFLAGS = -fPIC -Wextra -O2 -g -std=c++11			# C flags
LDFLAGS = -shared  							# linking flags
RM = rm -f  			# rm command
TARGET_LIB = libFahes.so # target lib

# SRCS = hello_world.cpp src1.c src2.c # source files
SRCS = $(shell ls *.cpp)
OBJS = $(SRCS:.cpp=.o)

.PHONY: all
all: ${TARGET_LIB}

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -o $@ $^

# $(SRCS:.cpp=.d):%.d:%.cpp
# 	$(CC) $(CXXFLAGS) -MM $< >$@

-include $(SRCS:.cpp=.d)

.PHONY: clean
clean:
	-${RM} ${TARGET_LIB} ${OBJS} $(SRCS:.cpp=.d)
rmo:
	-${RM} $(SRCS:.cpp=.o) $(SRCS:.cpp=.d)