#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>

#define MY_TAG 0


int isPerfectSquare(int number){
    int iVar;
    float fVar;
 
    fVar=sqrt((double)number);
    iVar=fVar;
 
    if(iVar==fVar)
        return 1;
    else
        return 0;
}



void printMatrix(int n, int **matrix){

    for (int row = 0; row < n; row++){
        printf("\n");
        for (int col = 0; col < n; col++){
            if(matrix[row][col] >= 9999)
                printf("- ");
            else
                printf("%d ", matrix[row][col]);
        }
    }
    printf("\n");
}

int **allocateNewMatrix(int n){
    int **board = (int **)malloc(n * sizeof(int *)); 

    // for each row allocate Cols ints
    for (int row = 0; row < n; row++) {
        board[row] = (int *)malloc(n * sizeof(int));
    }
    return board;
}

int **specialMatrixMultiply(int n, int **weightMatrix){

    // allocate Rows rows, each row is a pointer to int
    
    int **board = allocateNewMatrix(n);

    for (int row = 0; row < n; row++){
        for (int col = 0; col < n; col++){
            if(row != col)
                board[row][col] = 9999;
            else
                board[row][col] = 0;

            for (int k = 0; k < n; k++){
                
                if(weightMatrix[row][k] + weightMatrix[k][col] < board[row][col] && row != col){

                    board[row][col] = weightMatrix[row][k] + weightMatrix[k][col];       
                }

            }
        }
    }

    return board;
}


int **repeatedSquaringMethod(int n, int **weightMatrix){
    int m = 1;
    int **auxMatrix = allocateNewMatrix(n);
    
    while(m < (n-1)){

        if(m == 1){
            auxMatrix = specialMatrixMultiply(n, weightMatrix);
        }
        else{
            auxMatrix = specialMatrixMultiply(n, auxMatrix);
        }
        m = 2*m;
    }
    return auxMatrix;
}



int main(int argc, char **argv) {
  
    int n, my_rank, n_procs;
    MPI_Status status;
    
    n=0;

    /**
     * Inputs
    */

    scanf("%d", &n);

    int **rootMatrix = allocateNewMatrix(n);
    for (int row = 0; row < n; row++){
        for (int col = 0; col < n; col++){
            scanf("%d", &rootMatrix[row][col]);
            if(rootMatrix[row][col] == 0 && row != col)
                rootMatrix[row][col] = 9999;
        }
    }
    

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);



    if (my_rank == 0){
        /**
         * Make sure the presented configuration is valid for FOX's algorithm
         * 1 - Nº of processes is a perfect square whose square root (Q) evenly devides n
         * 2 - n % Q = 0
        */
        if(!isPerfectSquare(n_procs) || n % (int) sqrt(n_procs) != 0){
            printf("ERROR: Invalid Configuration\n");
            MPI_Finalize();
            return 0;
        }
        
        

        int **newMatrix = allocateNewMatrix(n);
        newMatrix = repeatedSquaringMethod(n, rootMatrix);
        printMatrix(n, newMatrix);
    }



    //printf("\nMy rank:%d my work: %d\n", my_rank, my_rank+n);
    //printf("\nProcesses:%d \n", n_procs);


    MPI_Finalize();
}