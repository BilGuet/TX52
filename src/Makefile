all: main
    
main: main.cpp
	nvcc -w ComputeNeighbors.cu ComputeNormalVector.cu ComputeWUsingNumberNeighbor.cu GetEigenValues.cu GetParticleSpacing.cu KNearestNeighbors.cu main.cpp SolveEigenvalues.cpp -o main

clean:
	rm main.exe
	
