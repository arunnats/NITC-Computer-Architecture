#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define TINY 1e-20
#define MAX_SIZE 100 // Maximum size for the matrix
#define a(i,j) a[(i)*N+(j)] // Access macro for matrix

void h_pivot_decomp(float *a, int *p, int *q, int N){
    int i,j,k;
    int pi,pj,tmp;
    float max;
    float ftmp;
    for (k=0;k<N;k++){
        pi=-1,pj=-1,max=0.0;
        //find pivot in submatrix a(k:N,k:N)
        for (i=k;i<N;i++) {
            for (j=k;j<N;j++) {
                if (fabs(a(i,j))>max){
                    max = fabs(a(i,j));
                    pi=i;
                    pj=j;
                }
            }
        }
        // Swap Row
        tmp=p[k];
        p[k]=p[pi];
        p[pi]=tmp;
        for (j=0;j<N;j++){
            ftmp=a(k,j);
            a(k,j)=a(pi,j);
            a(pi,j)=ftmp;
        }
        // Swap Col
        tmp=q[k];
        q[k]=q[pj];
        q[pj]=tmp;
        for (i=0;i<N;i++){
            ftmp=a(i,k);
            a(i,k)=a(i,pj);
            a(i,pj)=ftmp;
        }
        //END PIVOT

        //check pivot size and decompose
        if ((fabs(a(k,k))>TINY)){
            for (i=k+1;i<N;i++){
                //Column normalisation
                ftmp=a(i,k)/=a(k,k);
                for (j=k+1;j<N;j++){
                    //a(ik)*a(kj) subtracted from lower right submatrix elements
                    a(i,j)-=(ftmp*a(k,j));
                }
            }
        }
    }
}

void h_solve(float *a, float *x, int *p, int *q, int N){
    int i,ii=0,ip,j,tmp;
    float ftmp;
    float xtmp[MAX_SIZE]; // Changed to use MAX_SIZE
    // Swap rows (x=Px)
    puts("x=Px Stage");
    for (i=0; i<N; i++){
        xtmp[i]=x[p[i]]; //value that should be here
        printf("x:%.17lf\n", xtmp[i]); // Increased precision
    }
    // Lx=x
    puts("Lx=x Stage");
    for (i=0;i<N;i++){
        ftmp=xtmp[i];
        if (ii != 0)
            for (j=ii-1;j<i;j++)
                ftmp-=a(i,j)*xtmp[j];
        else
            if (ftmp!=0.0)
                ii=i+1;
        xtmp[i]=ftmp;
        printf("x:%.17lf\n", xtmp[i]);
    }
    puts("Ux=x");
    
    //backward substitution
    //solves Uy=z
    xtmp[N-1]/=a(N-1,N-1);
    for (i=N-2;i>=0;i--){
        ftmp=xtmp[i];
        for (j=i+1;j<N;j++){
            ftmp-=a(i,j)*xtmp[j];
        }
        xtmp[i]=(ftmp)/a(i,i);
    }
    for (i=0;i<N;i++)
        printf("x:%.17lf\n", xtmp[i]);

    // Last bit
    //solves x=Qy
    puts("x=Qx Stage");
    for (i=0;i<N;i++){
        x[i]=xtmp[q[i]];
        printf("x:%.17lf\n", x[i]);
    }
}

int main() {
    int N;
    float *a, *b;
    int *p_pivot, *q_pivot;

    // Read input from the file
    FILE *file = fopen("input.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return EXIT_FAILURE;
    }

    // Read the size of the system
    fscanf(file, "%d", &N);
    if (N > MAX_SIZE) {
        fprintf(stderr, "Matrix size exceeds maximum allowed size (%d).\n", MAX_SIZE);
        fclose(file);
        return EXIT_FAILURE;
    }

    // Allocate memory for matrix A and vector B
    a = (float *)malloc(N * N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    p_pivot = (int *)malloc(N * sizeof(int));
    q_pivot = (int *)malloc(N * sizeof(int));

    // Read matrix A
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            fscanf(file, "%f", &a(i,j));

    // Read vector B
    for (int i = 0; i < N; i++)
        fscanf(file, "%f", &b[i]);

    fclose(file);

    // Initialize pivot arrays
    for (int i = 0; i < N; i++) {
        p_pivot[i] = i;
        q_pivot[i] = i;
    }

    puts("Starting Stuff");
    h_pivot_decomp(a, p_pivot, q_pivot, N);
    puts("After Pivot");
    
    h_solve(a, b, p_pivot, q_pivot, N);
    puts("Finished Solve");

    // Free allocated memory
    free(a);
    free(b);
    free(p_pivot);
    free(q_pivot);

    return EXIT_SUCCESS;
}
