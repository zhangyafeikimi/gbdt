CC = gcc
CXX = g++
AR = ar
RANLIB = ranlib
CPPFLAGS = -Irapidjson-0.11/include
CFLAGS = -Wall -g -O3
CXXFLAGS = $(CFLAGS)
LIBS =

all: libgbdt.a gbdt-train gbdt-predict gbdt-benchmark

libgbdt.a: src/gain.o src/gbdt.o src/param.o src/sample.o src/x.o
	$(AR) -rc $@ $?
	$(RANLIB) $@

gbdt-train: src/gbdt-train.o libgbdt.a
	$(CXX) $(LIBS) -o $@ $?

gbdt-predict: src/gbdt-predict.o libgbdt.a
	$(CXX) $(LIBS) -o $@ $?

gbdt-benchmark: src/gbdt-benchmark.o libgbdt.a
	$(CXX) $(LIBS) -o $@ $?

%.o: %.cc
	$(CXX) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	rm -f src/*.o *.o *.a *.exe gbdt-train gbdt-predict gbdt-benchmark
