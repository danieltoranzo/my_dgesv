/****************************************************************************
*                                                                           *
*   Herramientas HPC (HPC Tools) - Assignment 2                             *
*                                                                           *
*   Daniel Toranzo Pérez                                                    *
*                                                                           *
*   dgesv.c                                                                 *
*                                                                           *
*   Uuna implementación para resolver sistemas de ecuaciones lineales       *
*   realizando la eliminación                                               *
*   Parámetros: #n                                                          *
*   #n: (int) Orden de la matriz > 0                                        *
*                                                                           *
****************************************************************************/

/**
 * @file dgesv.c
 * @author Daniel Toranzo Pérez
 * @brief Una implementación de la eliminación gaussiana.
 */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/**
* @brief Calcula el valor absoluto.
* @param value Valor.
* @return Devuelve Valor absoluto del valor.
*/
float absValue(const float value){
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
float *generate_matrix(const int size)
{
    int i;
    #ifdef __INTEL_COMPILER
    	float *matrix = _mm_malloc(sizeof(float)*size*size,32);
    	__assume_aligned(matrix,32);
    #else
		float *matrix = aligned_alloc(32,sizeof(float)*size*size);
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
void print_matrix(float *matrix, const int size)
{
    int i, j;
    printf("matrix: \n");
    #ifdef __INTEL_COMPILER
	    __assume_aligned(matrix,32);
	#else
	    matrix = __builtin_assume_aligned(matrix,32);
	#endif
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            printf("%.f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

/*
* @brief Función para intercambiar dos filas de una matriz de posición.
* @param matrix Matriz.
* @param row Fila que se va a intercambiar.
* @param row_new Fila que se va a intercambiar.
* @param n Tamaño de la fila.
*/
void swapRowFromMatrix(float *matrix, const int row, const int row_new, const int n){
    #ifdef __INTEL_COMPILER
    	__assume_aligned(matrix,32);
	#else
    	matrix = __builtin_assume_aligned(matrix,32);
	#endif
    for (int i=0;i<n;i++){
    	float temp = matrix[row*n +i];
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
void  substractRowFromWeightRow(float * matrix, const int row_result, const int row, const float weight, const int n){
    #ifdef __INTEL_COMPILER
    	__assume_aligned(matrix,32);
	#else
    	matrix = __builtin_assume_aligned(matrix,32);
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
int gaussianElimination(float* matrix, float *augmented_matrix,const int n){
	#ifdef __INTEL_COMPILER
	    __assume_aligned(matrix,32);
	    __assume_aligned(augmented_matrix,32);
	#else
	    matrix = __builtin_assume_aligned(matrix,32);
	    augmented_matrix = __builtin_assume_aligned(augmented_matrix,32);
	#endif
	int i=0;
    for (i = 0; i<n; i++) {
        if(matrix[i*n +i] == 0) {
            int greater = i;
            for (int j = 0; j < n; j++){
                float actual = matrix[j*n +i];
                float actual_max = matrix[greater*n +i];
                if (absValue(actual) > absValue(actual_max)){
                    greater = j;
                } 
            }
            if(greater != i){
                swapRowFromMatrix(matrix,i,greater,n);
                swapRowFromMatrix(augmented_matrix,i,greater,n);
            }else{
            	return -1;
            }
        }
   		for (int j=0;j<n;j++) {
   			if(i!=j){
				float weight = matrix[j*n +i]/matrix[i*n +i];
				if (weight != 0) {
					substractRowFromWeightRow(matrix,j,i,weight,n);
                	substractRowFromWeightRow(augmented_matrix,j,i,weight,n);
                }
			}
   		}
    }
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            augmented_matrix[i*n + j] /= matrix[i*n +i];
        }
    }
    return 0;
}

/*
* @brief Calcula X de AX=B y guarda el resultado en B.
* @param n Orden de la matriz.
* @param a Matriz.
* @param b Matriz con el resultado.
* @warning Funciona si la Mátriz no es singular.
*/
int my_dgesv(const int n, float *a, float *b) {
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
    float *a;
    float *b;
    a = generate_matrix(size);     
    b = generate_matrix(size);
    double tStart = omp_get_wtime();
    my_dgesv(size, a, b);
    printf("Time taken by my implementation: %.2fs\n", (omp_get_wtime() - tStart));
    //print_matrix(b, size);
    #ifdef __INTEL_COMPILER
    	_mm_free(a);
    	_mm_free(b);
    #else
		free(a);
    	free(b);
	#endif
}
