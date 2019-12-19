all: main
    
main: main.cpp Point.cpp KNearestNeighbors.cu
	nvcc -w SolveEigenvalues.cpp ComputeNeighbors.cu KNearestNeighbors.cu ComputeWUsingNumberNeighbor.cu main.cpp -o main

clean:
	rm main.exe
	
