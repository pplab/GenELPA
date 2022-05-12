# for taishan nodes of hanhai20

FLAGS=-Wall -mtune=native -march=armv8.2-a -Ofast -std=c++11
LIBS_legacy=-Llib -lelpa_legacy -L/home/nic/yshen/program/aarch64/elpa/2016.05.004/lib -lelpa
LIBS_new=-Llib -lelpa_new -L/home/nic/yshen/program/aarch64/elpa/2021.11.001/lib -lelpa
LIBS=-L/opt/software/scalapack/2.1.0/lib -lscalapack \
     -L/opt/software/OpenBLAS/0.3.10/lib -lopenblas \
     -lmpi_mpifh -lgfortran

INCLUDE_legacy=-I/home/nic/yshen/program/aarch64/elpa/2016.05.004/include/elpa-2016.05.004
INCLUDE_new=-I/home/yshen/program/elpa/2021.11.002/intel2020/include/elpa-2021.11.002
INCLUDE=-Iinclude

bin/test_double_new: test_double.o lib/libelpa_new.so
	mpicxx test_double.o ${FLAGS} ${LIBS_new} ${LIBS} -o bin/test_double_new

bin/test_double_legacy: test_double.o lib/libelpa_legacy.so
	mpicxx test_double.o ${FLAGS} ${LIBS_legacy} ${LIBS} -o bin/test_double_legacy

test_double.o: src/test_double.cpp
	mpicxx ${FLAGS} ${INCLUDE} -c src/test_double.cpp

lib/libelpa_legacy.so: elpa_legacy.o
	mpicxx ${FLAGS} -shared -o lib/libelpa_legacy.so elpa_legacy.o

elpa_legacy.o: src/elpa_legacy.cpp
	mpicxx ${FLAGS} ${INCLUDE} ${INCLUDE_legacy} -c -fPIC src/elpa_legacy.cpp

lib/libelpa_new.so: elpa_new.o
	mpicxx ${FLAGS} -shared -o lib/libelpa_new.so elpa_new.o

elpa_new.o: src/elpa_new.cpp
	mpicxx ${FLAGS} ${INCLUDE} ${INCLUDE_new} -c -fPIC src/elpa_new.cpp

clean:
	rm *.o lib/*.so bin/*
