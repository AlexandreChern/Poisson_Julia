#include <iostream>
#include <vector>
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <chrono>

int main()
{
    MPI_Init(NULL, NULL);

    int n, id;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    const size_t size_tot = 1024*1024*1024;
    const size_t size_max = size_tot / n;

    // CPU TEST
    std::vector<double> a_cpu_in (size_tot);
    std::vector<double> a_cpu_out(size_tot);
    std::fill(a_cpu_in.begin(), a_cpu_in.end(), id);

    std::cout << id << ": Starting CPU all-to-all\n";
    auto time_start = std::chrono::high_resolution_clock::now();
    MPI_Alltoall(
            a_cpu_in .data(), size_max, MPI_DOUBLE,
            a_cpu_out.data(), size_max, MPI_DOUBLE,
            MPI_COMM_WORLD);
    auto time_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(time_end-time_start).count();
    std::cout << id << ": Finished CPU all-to-all in " << std::to_string(duration) << " (ms)\n";

    // GPU TEST
    int id_local = id % 4;
    cudaSetDevice(id_local);
    double* a_gpu_in;
    double* a_gpu_out;
    cudaMalloc((void **)&a_gpu_in , size_tot * sizeof(double));
    cudaMalloc((void **)&a_gpu_out, size_tot * sizeof(double));
    cudaMemcpy(a_gpu_in, a_cpu_in.data(), size_tot*sizeof(double), cudaMemcpyHostToDevice);

    int id_gpu;
    cudaGetDevice(&id_gpu);
    std::cout << id << ", " << id_local << ", " << id_gpu << ": Starting GPU all-to-all\n";
    time_start = std::chrono::high_resolution_clock::now();
    MPI_Alltoall(
            a_gpu_in , size_max, MPI_DOUBLE,
            a_gpu_out, size_max, MPI_DOUBLE,
            MPI_COMM_WORLD);
    time_end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(time_end-time_start).count();

    std::cout << id << ", " << id_local << ", " << id_gpu << ": Finished GPU all-to-all in " << std::to_string(duration) << " (ms)\n";

    MPI_Finalize();
    return 0;
}
