#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#define MY_TAG 0
#define ROOT 0

#define INF 9999

typedef struct{
    int nProcesses;     /* Total number of processes  */
    MPI_Comm comm;      /* Communicator for entire grid */
    MPI_Comm row_comm;  /* Communicator for my row */
    MPI_Comm col_comm;  /* Communicator for my col */
    int q;              /* Order of grid */
    int my_row;         /* My Row number */
    int my_col;         /* My column number */
    int my_rank;        /* My RANK IN THE GRID COMMUNICATOR */

} GRID_INFO_TYPE;

GRID_INFO_TYPE *newGrid(){
    GRID_INFO_TYPE *new_grid_info = (GRID_INFO_TYPE *) malloc(sizeof(GRID_INFO_TYPE));
    return new_grid_info;
}

int **allocarray(int n) {
    int *data = malloc(n*n*sizeof(int));
    int **arr = malloc(n*sizeof(int *));
    for (int i=0; i<n; i++)
        arr[i] = &(data[i*n]);

    return arr;
}

void setupCommunicatorGrid(GRID_INFO_TYPE *grid, int old_rank, int n_procs){

    int dims[2], periods[2], coords[2], varying_coords[2];

    /*Global Information*/
    grid->nProcesses = n_procs;
    grid->q = (int) sqrt((double) grid->nProcesses);
    dims[0] = dims[1] = grid->q;
    periods[0] = periods[1] = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &(grid->comm));

    MPI_Comm_rank(grid->comm, &(grid->my_rank));
    MPI_Cart_coords(grid->comm, grid->my_rank, 2, coords);
    grid->my_row = coords[0];
    grid->my_col = coords[1];

    /*Set up row and column communicators*/
    varying_coords[0] = 0;
    varying_coords[1] = 1;
    MPI_Cart_sub(grid->comm, varying_coords, &(grid->row_comm));

    varying_coords[0] = 1;
    varying_coords[1] = 0;
    MPI_Cart_sub(grid->comm, varying_coords, &(grid->col_comm));
}

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
            if(matrix[row][col] >= INF)
                printf("0 ");
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
    
    int **board = allocarray(n);

    for (int row = 0; row < n; row++){
        for (int col = 0; col < n; col++){
            if(row != col)
                board[row][col] = INF;
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
    int **auxMatrix = allocarray(n);
    
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
  
    int n, my_world_rank, n_procs;
    double start, finish;
    MPI_Status status;
    
    n=0;


    /**
     * Inputs
    */

    scanf("%d", &n);

    int **rootMatrix = allocarray(n);
    for (int row = 0; row < n; row++){
        for (int col = 0; col < n; col++){
            scanf("%d", &rootMatrix[row][col]);
            if(rootMatrix[row][col] == 0 && row != col)
                rootMatrix[row][col] = INF;
        }
    }
    

    /**
     * Start Computations
    */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    MPI_Datatype submatrix;

    /**
     * Make sure the presented configuration is valid for FOX's algorithm
     * 1 - Nº of processes is a perfect square whose square root (Q) evenly devides n
     * 2 - n % Q = 0
    */
    //if (my_world_rank == 0){
        if(!isPerfectSquare(n_procs) || n % (int) sqrt(n_procs) != 0){
            printf("ERROR: Invalid Configuration\n");
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            return 1;
        }
    //}

    //Sequencial Repeated Squaring Method
    if(n_procs == 1){
        int **newMatrix = allocarray(n);
        newMatrix = repeatedSquaringMethod(n, rootMatrix);
        printMatrix(n, newMatrix);
    }else{

        //Parallel

        MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

        //TODO: apply FOx's algorithm to the matrix multiplication functions

        /**
         * Set Grid Communication topology
        */
        GRID_INFO_TYPE *grid = newGrid();
        setupCommunicatorGrid(grid, my_world_rank, n_procs);

        /**
         * 
         * Start: Handle sub-matrix distribution
         * 
        */

        int **local_A = allocarray((int) n/grid->q);
        int **local_B = allocarray((int) n/grid->q);
        int **local_C = allocarray((int) n/grid->q);

        int rootMatrixSizes[2], subMatrixSizes[2], subMatrixStart[2];

        if(my_world_rank == 0){
            rootMatrixSizes[0] = rootMatrixSizes[1] = n;
            subMatrixSizes[0] = subMatrixSizes[1] = (int) n/grid->q;

            int testprocs=0;

            for (int row = 0; row < n; row+=subMatrixSizes[0]){
                for (int col = 0; col < n; col+=subMatrixSizes[0]){
                    subMatrixStart[0]=row;
                    subMatrixStart[1]=col;

                    if(row == 0 && col ==0){ 
                        /*Get my submatrix*/
                        for(int auxRow = 0; auxRow<subMatrixSizes[0]; auxRow++){
                            for(int auxCol = 0; auxCol<subMatrixSizes[0]; auxCol++){
                                local_A[row][col] = rootMatrix[auxRow][auxCol];
                            }
                        }

                    }else{
                        //printf("I am WORLD process %d, sent: (%d,%d) to process %d\n", my_world_rank, subMatrixStart[0], subMatrixStart[1], testprocs);
                        
                        MPI_Type_create_subarray(2, rootMatrixSizes, subMatrixSizes, subMatrixStart, MPI_ORDER_C, MPI_INT, &submatrix);
                        MPI_Type_commit(&submatrix);
                        MPI_Send(&(rootMatrix[0][0]), 1, submatrix, testprocs, MY_TAG, MPI_COMM_WORLD);
                        MPI_Type_free(&submatrix);
                    }

                    testprocs++;
                }
            }

        }else if(my_world_rank != 0){
            rootMatrixSizes[0] = rootMatrixSizes[1] = n;
            subMatrixSizes[0] = subMatrixSizes[1] = (int) n/grid->q;
            MPI_Recv(&(local_A[0][0]), subMatrixSizes[0]*subMatrixSizes[0], MPI_INT, 0, MY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("I am WORLD process %d, submatrix received\n", my_world_rank);
            printMatrix(subMatrixSizes[0], local_A);
        }

        /**
         * 
         * End: Handle sub-matrix distribution
         * 
        */


       /**
         * 
         * Start: FoX Algorithm
         * 
        */




       /**
         * 
         * End: FoX Algorithm
         * 
        */


    }



    MPI_Barrier(MPI_COMM_WORLD);
    finish = MPI_Wtime();

    if (my_world_rank == 0)
        printf("Execution time: %f seconds\n", finish - start);

    
    MPI_Finalize();
}