/****************************************************************************
*                                                                           *
*   Herramientas HPC (HPC Tools) - Assignment 3                             *
*                                                                           *
*   Daniel Toranzo Pérez                                                    *
*                                                                           *
*   dgesv.c                                                                 *
*                                                                           *
*   Compara el tiempo de ejeucución de LAPACKE_dgesv y una implementación   *
*   realizando la eliminación gaussiana para resolver sistemas de           *
*   ecuaciones lineales                                                     *
*   Parámetros: #n                                                          *
*   #n: (int) Orden de la matriz > 0                                        *
*                                                                           *
****************************************************************************/

/**
 * @file dgesv.c
 * @author Daniel Toranzo Pérez
 * @date 15 Nov 2020
 * @brief Compara el tiempo de ejecución de LAPACKE_dgesv y una implementación realizando la eliminación gaussiana.
 */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
//#include "mkl_lapacke.h"
#include <omp.h>

/**
* @brief Calcula el valor absoluto.
* @param value Valor.
* @return Devuelve Valor absoluto del valor.
*/
double absValue(double value){
	if (value < 0){
		return -value;
	}else{
		return value;
	}
}

/**
* @brief Función para inicializar las matrices.
* @param size Orden de la matriz.
* @return Devuelve la matriz inicializada.
*/
double *generate_matrix(const int size)
{
    int i;
    #ifdef __INTEL_COMPILER
        double *matrix = _mm_malloc(sizeof(double)*size*size,32);
        __assume_aligned(matrix,32);
    #else
        double *matrix = aligned_alloc(32,sizeof(double)*size*size);
        matrix = __builtin_assume_aligned(matrix,32);
    #endif
    srand(1);
    for (i = 0; i < size * size; i++)
        matrix[i] = rand() % 100;
    return matrix;
}

/*
* @brief Función para imprimir por pantalla las matrices.
* @param name Nombre.
* @param matrix Matriz.
* @param size Orden de la matriz.
*/
void print_matrix(const char *name, double *matrix, int size)
{
    int i, j;
    printf("matrix: %s \n", name);
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            printf("%f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

/*
* @brief Función para comprobar los resultados.
* @param bref Matriz de referencia.
* @param b Matriz.
* @param size Orden de la matriz.
* @return (int) -1 en caso de error e (int) 0 en caso de éxito.
*/
int check_result(double *bref, double *b, int size) {
    int i;
    double tolerance = 0.0001;
    for(i=0;i<size*size;i++) {
    	double diff = bref[i] - b[i];
        if (absValue(diff) > tolerance) return 0;
    }
    return 1;
}

/*
* @brief Función para intercambiar dos filas de una matriz de posición.
* @param matrix Matriz.
* @param row Fila que se va a intercambiar.
* @param row_new Fila que se va a intercambiar.
* @param n Tamaño de la fila.
*/
void swapRowFromMatrix(double *matrix, const int row, const int row_new, const int n){
    #ifdef __INTEL_COMPILER
        __assume_aligned(matrix,32);
        #pragma vector aligned
        #pragma ivdep
    #else
        matrix = __builtin_assume_aligned(matrix,32);
        #pragma GCC ivdep
    #endif
    for (int i=0;i<n;i++){
        double temp = matrix[row*n +i];
        matrix[row*n +i] = matrix[row_new*n +i];
        matrix[row_new*n +i] = temp;
    }
}

/*
* @brief Función para restar a una fila otra fila de una matriz multiplicada por su peso.
* @param matrix Matriz.
* @param row Fila que se va a intercambiar.
* @param row_new Fila que se va a intercambiar.
* @param n Tamaño de la fila.
*/
void  substractRowFromWeightRow(double * matrix, const int row_result, const int row, const double weight, const int n){
    #ifdef __INTEL_COMPILER
        __assume_aligned(matrix,32);
        #pragma ivdep
        #pragma vector aligned
    #else
        matrix = __builtin_assume_aligned(matrix,32);
        #pragma GCC ivdep
    #endif
    for(int i=0;i<n;i++)
        matrix[row_result*n +i] -= weight*matrix[row*n +i];
}


/*
* @brief Realiza la eliminación gaussiana.
* @param matrix Matriz.
* @param augmented_matrix Matriz aumentada.
* @param n Orden de la matriz.
* @return (int) -1 en caso de error e (int) 0 en caso de éxito.
* @warning Funciona si la Mátriz no es singular.
*/
int gaussianElimination(double* matrix, double *augmented_matrix,const int n){
    #ifdef __INTEL_COMPILER
        __assume_aligned(matrix,32);
        __assume_aligned(augmented_matrix,32);
    #else
        matrix = __builtin_assume_aligned(matrix,32);
        augmented_matrix = __builtin_assume_aligned(augmented_matrix,32);
    #endif
    int return_code=0;
    #pragma omp parallel
    {
        for (int i = 0; i<n; i++) {
	    #pragma omp single
	    {
            if(matrix[i*n +i] == 0) {
                int greater = i;
                for (int j = 0; j < n; j++){
                    double actual = matrix[j*n +i];
                    double actual_max = matrix[greater*n +i];
                    if (absValue(actual) > absValue(actual_max)){
                        greater = j;
                    } 
                }
                if(greater != i){
                    swapRowFromMatrix(matrix,i,greater,n);
                    swapRowFromMatrix(augmented_matrix,i,greater,n);
                }else{
                    return_code=-1;
                }
            }
            }
            #pragma omp for schedule(static)
            for (int j=0;j<n;j++) {
                if(i!=j){
                    double weight = matrix[j*n +i]/matrix[i*n +i];
                    if (weight != 0) {
                        substractRowFromWeightRow(matrix,j,i,weight,n);
                        substractRowFromWeightRow(augmented_matrix,j,i,weight,n);
                    }
                }
            }
        }
        #pragma omp for schedule(static)
        for(int i=0; i<n; i++){
            #ifdef __INTEL_COMPILER
                #pragma vector aligned
                #pragma ivdep
            #else
                #pragma GCC ivdep
            #endif
            for(int j=0; j<n; j++){
                augmented_matrix[i*n + j] /= matrix[i*n +i];
            }
        }
    }
    return return_code;
}

/*
* @brief Calcula X de AX=B y guarda el resultado en B.
* @param n Orden de la matriz.
* @param a Matriz.
* @param b Matriz con el resultado.
* @warning Funciona si la Mátriz no es singular.
*/
int my_dgesv(const int n, double *a, double *b) {
    #ifdef __INTEL_COMPILER
        __assume_aligned(a,32);
        __assume_aligned(b,32);
    #else
        a = __builtin_assume_aligned(a,32);
        b = __builtin_assume_aligned(b,32);
    #endif
    if(gaussianElimination(a,b,n) == -1) {
        printf("Matrix is singular\n");
        printf("Couldn't perform Gaussian Elimination! \n");
        return -1;
    }
    return 0;
}

int main(int argc, char *argv[]) {
    int size = atoi(argv[1]);
    double *a;
    double *b; 
    //double *aref;
    //double *bref;
    a = generate_matrix(size);
    //aref = generate_matrix(size);  
    b = generate_matrix(size);
    //bref = generate_matrix(size);
    // Using MKL to solve the system
    /*
    MKL_INT n = size, nrhs = size, lda = size, ldb = size, info;
    MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*size);  
    clock_t tStart = clock();
    info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
    printf("Time taken by MKL: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
    */
    double start = omp_get_wtime();
    my_dgesv(size, a, b);
    printf("Time taken by my implementation: %.2fs\n", omp_get_wtime() - start);
    /*if (check_result(bref,b,size)==1)
        printf("Result is ok!\n");
    else    
        printf("Result is wrong!\n");
    */
    //print_matrix("X", bref, size);
    //print_matrix("X", b, size);
    #ifdef __INTEL_COMPILER
        _mm_free(a);
        _mm_free(b);
        //_mm_free(aref);
        //_mm_free(bref);
    #else
        free(a);
        free(b);
        //free(aref);
        //free(bref);
    #endif
    //free(ipiv);
}
