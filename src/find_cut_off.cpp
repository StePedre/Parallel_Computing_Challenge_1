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

#include <chrono>

#include <omp.h>


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

long indexBinarySearch(int *in, long begin, long end, int value) {
    long left = begin;
    long right = end;
    
    while (left < right) {
        long mid = (left + right) / 2;

        if (in[mid] == value)
            return mid;

        if (in[mid] < value)
            left = mid+1;
        else
            right = mid;
    }
    
    return left;
}

void MsMergeParallelize(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin, int depth, int cutoff) {
	if((end1 - begin1) + (end2 - begin2) > cutoff){
		const long half1 = (begin1 + end1) / 2;
		long half2 = indexBinarySearch(in, begin2, end2, in[half1]);
		long outBegin2 = outBegin + (half1 - begin1) + (half2 - begin2);
        
        #pragma omp task
            MsMergeParallelize(out, in, begin1, half1, begin2, half2, outBegin, depth +1, cutoff);
        #pragma omp task
            MsMergeParallelize(out, in, half1, end1, half2, end2, outBegin2, depth + 1, cutoff);
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


int main(int argc, char* argv[]){
    // variables to measure the elapsed time
	struct timeval t1, t2, t3, t4;
	double etime;

	// expect one command line arguments: array size
	if (argc != 2) {
		printf("Usage: find_cut_off.exe <array size> \n");
		printf("\n");
		return EXIT_FAILURE;
	}else{
        int cutoff = 100;
        const size_t stSize = strtol(argv[1], NULL, 10);
        int *in = (int*) malloc(stSize * sizeof(int));
        int *out = (int*) malloc(stSize * sizeof(int));

        for(int i = 0; i<stSize/2; i++){
            in[i] = 2*i;
        }
        for(int i = stSize/2; i<stSize; i++){
            in[i] = ((i-stSize/2) *2 ) + 1;
        }
        
        MsMergeSequential(out, in, 0, stSize/2, stSize/2, stSize, 0);

        auto startSequential = std::chrono::high_resolution_clock::now();
        MsMergeSequential(out, in, 0, stSize/2, stSize/2, stSize, 0);
        auto endSequential  = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> durationSequential = endSequential  - startSequential;

        auto startParallel = std::chrono::high_resolution_clock::now();;
        auto endParallel = std::chrono::high_resolution_clock::now();;
        std::chrono::duration<double, std::milli> durationParallel;
        #pragma omp parallel
        {
            #pragma omp single
            {
                MsMergeParallelize(out, in, 0, stSize/2, stSize/2, stSize, 0, 0, cutoff);          
            }
        }

        do{
            #pragma omp parallel
            {
                #pragma omp single
                {
                    startParallel = std::chrono::high_resolution_clock::now(); 
                    MsMergeParallelize(out, in, 0, stSize/2, stSize/2, stSize, 0, 0, cutoff);          
                    endParallel = std::chrono::high_resolution_clock::now();
                    durationParallel  = endParallel  - startParallel ;
                 
                }
            }
            printf("Cutoff (%d) - Sequential Time: %f - Parallel Time: %f \n", cutoff, durationSequential.count() ,durationParallel.count());
            if(durationParallel > durationSequential)
                cutoff+=100;
        }while(durationParallel > durationSequential);
        printf("%d \n", cutoff);
 
    }
}