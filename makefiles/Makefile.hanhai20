# for taishan nodes of hanhai20
MPICXX=mpiicpc

FLAGS=-Wall -mtune=native -march=native -Ofast -std=c++11 
LIBS_legacy=-Llib -lelpa_legacy -L/home/nic/yshen/program/elpa/2016.05.004/lib -lelpa
LIBS_new=-Llib -lelpa_new -L/home/nic/yshen/program/elpa/2021.11.002-A100/lib -lelpa
LIBS=-L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl

INCLUDE_legacy=-I/home/nic/yshen/program/elpa/2016.05.004/include/elpa-2016.05.004
INCLUDE_new=-I/home/nic/yshen/program/elpa/2021.11.002-A100/include/elpa-2021.11.002
INCLUDE=-Iinclude -I${MKLROOT}/include

bin/test_double_new: test_double.o lib/libelpa_new.so
	${MPICXX} test_double.o ${FLAGS} ${LIBS_new} ${LIBS} -o bin/test_double_new

bin/test_double_legacy: test_double.o lib/libelpa_legacy.so
	${MPICXX} test_double.o ${FLAGS} ${LIBS_legacy} ${LIBS} -o bin/test_double_legacy

bin/test_complex_new: test_complex.o lib/libelpa_new.so
	${MPICXX} test_complex.o ${FLAGS} ${LIBS_new} ${LIBS} -o bin/test_complex_new

bin/test_complex_legacy: test_complex.o lib/libelpa_legacy.so
	${MPICXX} test_complex.o ${FLAGS} ${LIBS_legacy} ${LIBS} -o bin/test_complex_legacy

test_double.o: src/test_double.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} -c src/test_double.cpp

test_complex.o: src/test_complex.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} -c src/test_complex.cpp

lib/libelpa_legacy.so: elpa_legacy.o elpa_legacy_real.o elpa_legacy_complex.o utils.o
	${MPICXX} ${FLAGS} -shared -o lib/libelpa_legacy.so elpa_legacy.o elpa_legacy_real.o elpa_legacy_complex.o utils.o

elpa_legacy.o: src/elpa_legacy.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} ${INCLUDE_legacy} -c -fPIC src/elpa_legacy.cpp

elpa_legacy_real.o: src/elpa_legacy_real.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} ${INCLUDE_legacy} -c -fPIC src/elpa_legacy_real.cpp

elpa_legacy_complex.o: src/elpa_legacy_complex.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} ${INCLUDE_legacy} -c -fPIC src/elpa_legacy_complex.cpp

lib/libelpa_new.so: elpa_new.o elpa_new_real.o elpa_new_complex.o utils.o
	${MPICXX} ${FLAGS} -shared -o lib/libelpa_new.so elpa_new.o elpa_new_real.o elpa_new_complex.o utils.o

elpa_new.o: src/elpa_new.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} ${INCLUDE_new} -c -fPIC src/elpa_new.cpp

elpa_new_real.o: src/elpa_new_real.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} ${INCLUDE_new} -c -fPIC src/elpa_new_real.cpp

elpa_new_complex.o: src/elpa_new_complex.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} ${INCLUDE_new} -c -fPIC src/elpa_new_complex.cpp

utils.o: src/utils.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} -c -fPIC src/utils.cpp

clean:
	rm *.o lib/*.so bin/*
