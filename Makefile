# use static lib for hanhai20
MPICXX=mpiicpc

ELPA_DIR_legacy=/home/nic/yshen/program/elpa/2016.05.004
ELPA_VER_legacy=elpa-2016.05.004
ELPA_DIR_new=/gpfs/opt/elpa/elpa-2021.05.002_intel2018
ELPA_VER_new=elpa_openmp-2021.05.002

ELPA_LIB_legacy=${ELPA_DIR_legacy}/lib
ELPA_INCLUDE_legacy=${ELPA_DIR_legacy}/include/${ELPA_VER_legacy}
ELPA_LIB_new=${ELPA_DIR_new}/lib
ELPA_INCLUDE_new=${ELPA_DIR_new}/include/${ELPA_VER_new}

FLAGS=-Wall -mtune=native -march=native -Ofast -std=c++11 
LIBS_legacy=-Llib -lelpa_legacy -L${ELPA_LIB_legacy} -lelpa
LIBS_new=-Llib -lelpa_new -L${ELPA_LIB_new} -lelpa_openmp
LIBS=-L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl

INCLUDE_legacy=-I${ELPA_INCLUDE_legacy}
INCLUDE_new=-I${ELPA_INCLUDE_new}
INCLUDE=-Iinclude -I${MKLROOT}/include

bin/test_multi_elpa_handle_double: test_multi_elpa_handle_double.o lib/libelpa_new.a
	${MPICXX} test_multi_elpa_handle_double.o ${FLAGS} ${LIBS_new} ${LIBS} -o bin/test_multi_elpa_handle_double

test_multi_elpa_handle.o: src/test_multi_elpa_handle.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} -c src/test_multi_elpa_handle.cpp

bin/test_multi_elpa_handle_complex: test_multi_elpa_handle_complex.o lib/libelpa_new.a
	${MPICXX} test_multi_elpa_handle_complex.o ${FLAGS} ${LIBS_new} ${LIBS} -o bin/test_multi_elpa_handle_complex

test_multi_elpa_handle_complex.o: src/test_multi_elpa_handle_complex.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} -c src/test_multi_elpa_handle_complex.cpp

bin/benchmark_double_new: benchmark_double.o lib/libelpa_new.a
	${MPICXX} benchmark_double.o ${FLAGS} ${LIBS_new} ${LIBS} -o bin/benchmark_double_new

bin/benchmark_double_legacy: benchmark_double.o lib/libelpa_legacy.a
	${MPICXX} benchmark_double.o ${FLAGS} ${LIBS_legacy} ${LIBS} -o bin/benchmark_double_legacy

bin/benchmark_complex_new: benchmark_complex.o lib/libelpa_new.a
	${MPICXX} benchmark_complex.o ${FLAGS} ${LIBS_new} ${LIBS} -o bin/benchmark_complex_new

bin/benchmark_complex_legacy: benchmark_complex.o lib/libelpa_legacy.a
	${MPICXX} benchmark_complex.o ${FLAGS} ${LIBS_legacy} ${LIBS} -o bin/benchmark_complex_legacy

benchmark_double.o: src/benchmark_double.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} -c src/benchmark_double.cpp

benchmark_complex.o: src/benchmark_complex.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} -c src/benchmark_complex.cpp

bin/test_double_new: test_double.o lib/libelpa_new.a
	${MPICXX} test_double.o ${FLAGS} ${LIBS_new} ${LIBS} -o bin/test_double_new

bin/test_double_legacy: test_double.o lib/libelpa_legacy.a
	${MPICXX} test_double.o ${FLAGS} ${LIBS_legacy} ${LIBS} -o bin/test_double_legacy

bin/test_complex_new: test_complex.o lib/libelpa_new.a
	${MPICXX} test_complex.o ${FLAGS} ${LIBS_new} ${LIBS} -o bin/test_complex_new

bin/test_complex_legacy: test_complex.o lib/libelpa_legacy.a
	${MPICXX} test_complex.o ${FLAGS} ${LIBS_legacy} ${LIBS} -o bin/test_complex_legacy

test_double.o: src/test_double.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} -c src/test_double.cpp

test_complex.o: src/test_complex.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} -c src/test_complex.cpp

lib/libelpa_legacy.a: elpa_legacy.o elpa_legacy_real.o elpa_legacy_complex.o utils.o
	ar rcsv lib/libelpa_legacy.a elpa_legacy.o elpa_legacy_real.o elpa_legacy_complex.o utils.o

elpa_legacy.o: src/elpa_legacy.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} ${INCLUDE_legacy} -c -fPIC src/elpa_legacy.cpp

elpa_legacy_real.o: src/elpa_legacy_real.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} ${INCLUDE_legacy} -c -fPIC src/elpa_legacy_real.cpp

elpa_legacy_complex.o: src/elpa_legacy_complex.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} ${INCLUDE_legacy} -c -fPIC src/elpa_legacy_complex.cpp

lib/libelpa_new.a: include/my_elpa_generic.hpp elpa_new.o elpa_new_real.o elpa_new_complex.o utils.o include/my_elpa_generic.hpp
	ar rcsv lib/libelpa_new.a elpa_new.o elpa_new_real.o elpa_new_complex.o utils.o

elpa_new.o: src/elpa_new.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} ${INCLUDE_new} -c -fPIC src/elpa_new.cpp

elpa_new_real.o: src/elpa_new_real.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} ${INCLUDE_new} -c -fPIC src/elpa_new_real.cpp

elpa_new_complex.o: src/elpa_new_complex.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} ${INCLUDE_new} -c -fPIC src/elpa_new_complex.cpp

include/my_elpa_generic.hpp: script/elpa_generic_template_1.hpp script/elpa_generic_template_1.hpp
	script/translate_elpa_generic.sh ${ELPA_INCLUDE_new}

utils.o: src/utils.cpp
	${MPICXX} ${FLAGS} ${INCLUDE} -c -fPIC src/utils.cpp

clean:
	rm *.o lib/* bin/* include/my_elpa_generic.hpp
