
all: dgesv

dgesv: dgesv.c
	icc -O2 -no-prec-div -fopenmp -o dgesv dgesv.c

run:
	echo "Small test"
	./dgesv 2048
	echo "Medium test"
	./dgesv 4096
	echo "Large test"
	./dgesv 8192

clean:
	rm dgesv
