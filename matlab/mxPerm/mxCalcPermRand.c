/* mxCalcPermRand - calculates a random permutation from image patches

[ sortedIx , randCount ] = mxCalcPermRand( x_1Vec , x_restVec , ixVec , N , pointsNum , picDim , picDim2, randStartIx , B , randVec , eps , regGrid );
calculates a random permutation from the image patches by 
 1. starting from the patch with the index randStartIx    
 2. continue from each patch to its unvisited nearest neighbor in a BxB search area surrounding it
=====================================================================================
Input:
x_1Vec, x_restVec - vectors used to calculate distances between patches.
ixVec - a vector containing the patch coordinates.
N - number of pixels in a patch + 2. 
pointsNum - number of patches to reorder.
picDim - number of rows in the image.
picDim2 - number of columns
randStartIx - the index of the first point in the nearest neighbor search.
B - number of rows in the square search area used for each patch.
randVec - random variables used to calculate the permutations.
eps - design paramters.
regGrid - grid type: set to one if the grid is regular, other wise set to zero.
=====================================================================================
Output:
sortedIx - the calculated permutation
randCount - the number of patches for which the second nearest neighbor was selected
=====================================================================================
Idan Ram
Department of Electrical Engineering
Technion, Haifa 32000 Israel
idanram@tx.technion.ac.il

November 2013
===================================================================================== */
        
#include "mex.h"
#include "math.h"

#define OK 0
#define ERROR -1
#define NARGIN 12
#define NARGOUT 2

int mxCalcPermRand(
        /* inputs */
        double * x_1Vec,
        double * x_restVec,
        double * ixVec,
        int N,
        int pointsNum,
        int picDim,
		int picDim2,
        int randStartIx,
        double B,
        double * randVec,
        double eps,
        int regGrid,
        
        /* outputs */
        int * sortedIx,
        int * randCount
        ) {
    
    int n , m , k , t , z , b_x , b_y , empty, empty2, B_h, tempMinIx , tempMinIx2;
    double tempMinVal , tempDist, tempMinVal2;
    int * freePoints;
    double * tempVec;
    
    freePoints = ( int * ) malloc( pointsNum * sizeof( int ) );
    if(freePoints==NULL) {
        return ERROR;
    }
    
    tempVec = ( double * ) malloc( N * sizeof( double ) );
    if(tempVec==NULL) {
        return ERROR;
    }
    
    for ( n = 0 ; n < pointsNum ; n++ ) {
        freePoints[n] = 0;
    }
    
    /* finding the nn path */
    
    B_h = ( int ) floor( B / 2 );
    sortedIx[0] = randStartIx;
    randCount[0] = 0;
    freePoints[ sortedIx[0] - 1 ] = 1;
    for ( k = 0 ; k < pointsNum - 1 ; k++ ) {
        
        for ( n = 0 ; n < N ; n++ ) {
            tempVec[n] = x_1Vec[ ( sortedIx[k] - 1 ) * N + n ];
        }
        empty = 0;
        empty2 = 0;
        if ( B < picDim ) {
            if ( regGrid == 1 ) {
                n = (int) floor( ( sortedIx[k] - 1 ) / picDim );//heng zuo biao
                m = ( sortedIx[k] - 1 ) % picDim;//zong zuo biao
                for ( b_x = -B_h ; b_x <= B_h ; b_x++ ) {
                    if ( n + b_x >=0 && n + b_x < picDim2  ){
                        for ( b_y = -B_h ; b_y <= B_h ; b_y++ ) {
                            if ( m + b_y >=0 && m + b_y < picDim  ){
                                t = ( n + b_x ) * picDim + m + b_y;
                                if ( freePoints[t] != 1 ){
                                    tempDist = 0;
                                    for ( z = 0 ; z < N ; z++ ) {
                                        tempDist = tempDist -2 * tempVec[z] * x_restVec[ t * N + z ];
                                    }
                                    if ( empty == 0 ) {
                                        empty = 1;
                                        tempMinVal = tempDist;
                                        tempMinIx = t + 1; /* incremented by 1 for matlab */
                                    }
                                    else if ( tempMinVal > tempDist ){
                                        empty2 = 1;
                                        tempMinIx2 = tempMinIx;
                                        tempMinVal2 = tempMinVal;
                                        tempMinVal = tempDist;
                                        tempMinIx = t + 1; /* incremented by 1 for matlab */
                                    }
                                    else if ( ( tempMinVal2 > tempDist ) | ( empty2 == 0 ) ) {
                                        empty2 = 1;
                                        tempMinVal2 = tempDist;
                                        tempMinIx2 = t + 1; /* incremented by 1 for matlab */
                                    }
                                    
                                }
                            }
                        }
                    }
                }
            }
            else {
                for ( m = 0 ; m < pointsNum ; m++ ) {
                    
                    if ( fabs( ixVec[m] - ixVec[ sortedIx[k] - 1 ] ) <= B / 2 && fabs( ixVec[m + pointsNum ] - ixVec[ sortedIx[k] - 1 + pointsNum ] ) <= B / 2 && freePoints[m] != 1 ) {
                        
                        tempDist = 0;
                        for ( n = 0 ; n < N ; n++ ) {
                            tempDist = tempDist -2 * tempVec[n] * x_restVec[ m * N + n ];
                        }
                        if ( empty == 0 ) {
                            empty = 1;
                            tempMinVal = tempDist;
                            tempMinIx = m + 1; /* incremented by 1 for matlab */
                        }
                        else if ( tempMinVal > tempDist ){
                            empty2 = 1;
                            tempMinIx2 = tempMinIx;
                            tempMinVal2 = tempMinVal;
                            tempMinVal = tempDist;
                            tempMinIx = m + 1; /* incremented by 1 for matlab */
                        }
                        else if ( ( tempMinVal2 > tempDist ) | ( empty2 == 0 ) ) {
                            empty2 = 1;
                            tempMinVal2 = tempDist;
                            tempMinIx2 = m + 1; /* incremented by 1 for matlab */
                        }
                    }
                }
            }
        }
        
        if ( empty == 0 ) {
            for ( m = 0 ; m < pointsNum ; m++ ) {
                
                if ( freePoints[m] != 1 ) {
                    
                    tempDist = 0;
                    for ( n = 0 ; n < N ; n++ ) {
                        tempDist = tempDist -2 * tempVec[n] * x_restVec[ m * N + n ];
                    }
                    if ( empty == 0 ) {
                        empty = 1;
                        tempMinVal = tempDist;
                        tempMinIx = m + 1;
                    }
                    else if ( tempMinVal > tempDist ){
                        empty2 = 1;
                        tempMinIx2 = tempMinIx;
                        tempMinVal2 = tempMinVal;
                        tempMinVal = tempDist;
                        tempMinIx = m + 1;
                    }
                    else if ( ( tempMinVal2 > tempDist ) | ( empty2 == 0 ) ) {
                        empty2 = 1;
                        tempMinVal2 = tempDist;
                        tempMinIx2 = m + 1;
                    } 
                }
            }
        }
        tempMinVal = 1/ ( 1 + exp( ( tempMinVal - tempMinVal2 ) / ( N - 2 ) / eps ) );
        
        if ( ( empty2 == 1 ) && ( k < pointsNum - 2 ) && ( randVec[ k + 1 ] > tempMinVal ) ) {
            sortedIx[k + 1] = tempMinIx2;
            randCount[0] = randCount[0] + 1;
        }
        else {
            sortedIx[k + 1] = tempMinIx;
        }
        /*
         * printf( " sortedIx = %d \n " , sortedIx[k] );
         */
        freePoints[sortedIx[k + 1]-1] = 1;
    }
    free(freePoints);
    free(tempVec);
    
    return OK;
    
}

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[] ) {
    /* inputs */
    double * x_1Vec;
    double * x_restVec;
    double * ixVec;
    int N;
    int pointsNum;
    int picDim;
	int picDim2;
    int randStartIx;
    double B;
    double * randVec;
    double eps;
    int regGrid;
    
    /* outputs */
    int * sortedIx;
    int * randCount;
    
    int dataSize;
    unsigned int dimsS[2];
    unsigned int dimsC[2];
    int ret_code;
    
    /* Check for proper number of arguments. */
    if(nrhs != NARGIN) {
        mexErrMsgTxt("9 inputs required.");
    } else if(nlhs > NARGOUT) {
        mexErrMsgTxt("Too many output arguments");
    }
    
    /* prhs - array of inputs */
    x_1Vec = mxGetPr(prhs[0]);
    x_restVec = mxGetPr(prhs[1]);
    ixVec = mxGetPr(prhs[2]);
    N = (int) *mxGetPr(prhs[3]);
    pointsNum = (int) *mxGetPr(prhs[4]);
    picDim = (int) *mxGetPr(prhs[5]);
	picDim2 = (int) *mxGetPr(prhs[6]);
    randStartIx = (int) *mxGetPr(prhs[7]);
    B = *mxGetPr(prhs[8]);
    randVec = mxGetPr(prhs[9]);
    eps = *mxGetPr(prhs[10]);
    regGrid = (int) *mxGetPr(prhs[11]);
    
    dimsS[0] = pointsNum;
    dimsS[1] = 1;
    dimsC[0] = 1;
    dimsC[1] = 1;
    
    /* plhs - array of outputs */
    plhs[0] = mxCreateNumericArray(2, dimsS, mxINT32_CLASS, mxREAL);
    sortedIx = mxGetData(plhs[0]);
    plhs[1] = mxCreateNumericArray(2, dimsC, mxINT32_CLASS, mxREAL);
    randCount = mxGetData(plhs[1]);
    
    /* call the calculating function */
    ret_code = mxCalcPermRand( x_1Vec , x_restVec , ixVec , N , pointsNum , picDim , picDim2, randStartIx , B , randVec , eps , regGrid , sortedIx , randCount );
    if (ret_code==ERROR) {
        mexErrMsgTxt("error occured");
    }
}



