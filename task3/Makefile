CFLAGS = -std=c++98 -Wall -Wextra -fopenmp
#CFLAGS += -I/usr/lib/gcc/x86_64-linux-gnu/9/include/
SOURCES = task3.cpp cube.cpp
GENERATES=task3

all: common.h ${SOURCES}
	mpicxx ${CFLAGS} ${SOURCES} -O3 -o task3

debug: common.h ${SOURCES}
	mpicxx ${CFLAGS} ${SOURCES} -g -DDEBUG -o task3

seq: seq.c
	gcc -std=c99 -Wall -Wextra -O3 seq.c -o seq -lm

seq_debug: seq.c
	gcc -std=c99 -Wall -Wextra -O3 -g -DDEBUG seq.c -o seq_debug -lm

clean:
	rm -rf ${GENERATES}