/*
*  This file is part of Christian's OpenMP software lab 
*
*  Copyright (C) 2016 by Christian Terboven <terboven@itc.rwth-aachen.de>
*  Copyright (C) 2016 by Jonas Hahnfeld <hahnfeld@itc.rwth-aachen.de>
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>

#include <iostream>
#include <algorithm>

#include <cstdlib>
#include <cstdio>

#include <cmath>
#include <ctime>
#include <cstring>

#include <omp.h>


/**
  * helper routine: check if array is sorted correctly
  */
bool isSorted(int ref[], int data[], const size_t size){
	std::sort(ref, ref + size);
	for (size_t idx = 0; idx < size; ++idx){
		if (ref[idx] != data[idx]) {
			return false;
		}
	}
	return true;
}


/**
  * sequential merge step (straight-forward implementation)
  */
// TODO: cut-off could also apply here (extra parameter?)
// TODO: optional: we can also break merge in two halves
void MsMergeSequential(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin) {
	long left = begin1;
	long right = begin2;

	long idx = outBegin;

	while (left < end1 && right < end2) {
		if (in[left] <= in[right]) {
			out[idx] = in[left];
			left++;
		} else {
			out[idx] = in[right];
			right++;
		}
		idx++;
	}

	while (left < end1) {
		out[idx] = in[left];
		left++, idx++;
	}

	while (right < end2) {
		out[idx] = in[right];
		right++, idx++;
	}
}

/**
  * sequential merge step (straight-forward implementation)
  */
// TODO: cut-off could also apply here (extra parameter?)
// TODO: optional: we can also break merge in two halves
void MsMergeParallelize(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin, int depth) {
	if(depth < ceil(log2(omp_get_max_threads())) && (end1 - begin1) + (end2 - begin2) > 1024){
		const long half1 = (begin1 + end1) / 2;
		long half2 = begin2;
		while(in[half1] > in[half2] && half2<end2)
			half2++;
		long outMid = outBegin + (half1 - begin1) + (half2 - begin2);
        
        #pragma omp task
            MsMergeParallelize(out, in, begin1, half1, begin2, half2, outBegin, depth +1);
        #pragma omp task
            MsMergeParallelize(out, in, half1, end1, half2, end2, outMid, depth + 1);
        #pragma omp taskwait
    }else{
        long left = begin1;
        long right = begin2;

        long idx = outBegin;

        while (left < end1 && right < end2) {
            if (in[left] <= in[right]) {
                out[idx] = in[left];
                left++;
            } else {
                out[idx] = in[right];
                right++;
            }
            idx++;
        }

        while (left < end1) {
            out[idx] = in[left];
            left++, idx++;
        }

        while (right < end2) {
            out[idx] = in[right];
            right++, idx++;
        }
    }
    
}

/**
  * sequential MergeSort
  */
// TODO: remember one additional parameter (depth)
// TODO: recursive calls could be taskyfied
// TODO: task synchronization also is required
void MsSequential(int *array, int *tmp, bool inplace, long begin, long end) {
	if (begin < (end - 1)) {
		const long half = (begin + end) / 2;
		MsSequential(array, tmp, !inplace, begin, half);
		MsSequential(array, tmp, !inplace, half, end);
		if (inplace) {
			MsMergeSequential(array, tmp, begin, half, half, end, begin);
		} else {
			MsMergeSequential(tmp, array, begin, half, half, end, begin);
		}
	} else if (!inplace) {
		tmp[begin] = array[begin];
	}
}

/**
  * sequential MergeSort
  */
// TODO: remember one additional parameter (depth)
// TODO: recursive calls could be taskyfied
// TODO: task synchronization also is required
void MsParallelize(int *array, int *tmp, bool inplace, long begin, long end, int depth) {
	if (begin < (end - 1)) {
		const long half = (begin + end) / 2;
        if(depth < ceil(log2(omp_get_max_threads()))){
            #pragma omp task  
		        MsParallelize(array, tmp, !inplace, begin, half, depth+1);
            #pragma omp task
                MsParallelize(array, tmp, !inplace, half, end, depth+1);
            #pragma omp taskwait
            if (inplace) {
			    MsMergeParallelize(array, tmp, begin, half, half, end, begin, depth);
            } else {	
            	MsMergeParallelize(tmp, array, begin, half, half, end, begin, depth);
            }
        }else{ 
		    MsSequential(array, tmp, !inplace, begin, half);
            MsSequential(array, tmp, !inplace, half, end);
            if (inplace) {
			    MsMergeSequential(array, tmp, begin, half, half, end, begin);
            } else {
                MsMergeSequential(tmp, array, begin, half, half, end, begin);
            }
        }	
	} else if (!inplace) {
		tmp[begin] = array[begin];
	}
}


/**
  * Serial MergeSort
  */
// TODO: this function should create the parallel region
// TODO: good point to compute a good depth level (cut-off)
void MsSerial(int *array, int *tmp, const size_t size) {

   // TODO: parallel version of MsSequential will receive one more parameter: 'depth' (used as cut-off)
	MsSequential(array, tmp, true, 0, size);
}

/**
  * Parallel MergeSort
  */
// TODO: this function should create the parallel region
// TODO: good point to compute a good depth level (cut-off)
void MsParallel(int *array, int *tmp, const size_t size) {

    #pragma omp parallel
        #pragma omp single
	        MsParallelize(array, tmp, true, 0, size, 0);
}


/** 
  * @brief program entry point
  */
int main(int argc, char* argv[]) {
	// variables to measure the elapsed time
	struct timeval t1, t2;
	double etime;

	// expect one command line arguments: array size
	if (argc != 2) {
		printf("Usage: MergeSort.exe <array size> \n");
		printf("\n");
		return EXIT_FAILURE;
	}
	else {
		const size_t stSize = strtol(argv[1], NULL, 10);
		int *data = (int*) malloc(stSize * sizeof(int));
		int *tmp = (int*) malloc(stSize * sizeof(int));
		int *ref = (int*) malloc(stSize * sizeof(int));

		printf("Initialization...\n");

		srand(95);
		for (size_t idx = 0; idx < stSize; ++idx){
			data[idx] = (int) (stSize * (double(rand()) / RAND_MAX));
		}
		std::copy(data, data + stSize, ref);

		double dSize = (stSize * sizeof(int)) / 1024 / 1024;
		printf("Sorting %zu elements of type int (%f MiB)...\n", stSize, dSize);

		gettimeofday(&t1, NULL);
		MsParallel(data, tmp, stSize);
		gettimeofday(&t2, NULL);

		etime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
		etime = etime / 1000;

		printf("done, took %f sec. Verification...", etime);
		if (isSorted(ref, data, stSize)) {
			printf(" successful.\n");
		}
		else {
			printf(" FAILED.\n");
		}

		free(data);
		free(tmp);
		free(ref);
	}

	return EXIT_SUCCESS;
}
