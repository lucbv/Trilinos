/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <cstdio>

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <limits.h>
#include <cmath>
#include <unordered_map>

#include <Kokkos_Core.hpp>
#include<KokkosSparse_spmv.hpp>
#include<KokkosKernels_Test_Structured_Matrix.hpp>

enum {STRUCT, UNSTR};
enum {AUTO, DYNAMIC, STATIC};

#ifdef INT64
typedef long long int LocalOrdinalType;
#else
typedef int LocalOrdinalType;
#endif

template<typename Scalar>
int test_crs_matrix_singlevec(const bool banner, const int algo,
                              const int nx, const int ny, const int leftBC, const int rightBC,
                              const int bottomBC, const int topBC, const int stencil_type,
                              const int rows_per_thread, const int team_size,
                              const int vector_length, const int schedule, const int loop) {
  typedef KokkosSparse::CrsMatrix<Scalar,int,Kokkos::DefaultExecutionSpace,void,int> matrix_type ;
  typedef typename Kokkos::View<Scalar*,Kokkos::LayoutLeft> mv_type;
  typedef typename Kokkos::View<Scalar*,Kokkos::LayoutLeft,Kokkos::MemoryRandomAccess > mv_random_read_type;
  typedef typename mv_type::HostMirror h_mv_type;

  Kokkos::View<int*[3], typename matrix_type::memory_space> mat_structure("Matrix Structure", 2);
  typename Kokkos::View<int*[3], typename matrix_type::memory_space>::HostMirror mat_structure_h
    = Kokkos::create_mirror_view(mat_structure);
  Kokkos::deep_copy(mat_structure_h, mat_structure);
  mat_structure_h(0, 0) = nx;
  mat_structure_h(1, 0) = ny;
  if(leftBC   == 1) { mat_structure_h(0, 1) = 1; }
  if(rightBC  == 1) { mat_structure_h(0, 2) = 1; }
  if(bottomBC == 1) { mat_structure_h(1, 1) = 1; }
  if(topBC    == 1) { mat_structure_h(1, 2) = 1; }
  Kokkos::deep_copy(mat_structure, mat_structure_h);

  std::string discrectization_stencil;
  if(stencil_type == 1) {
    discrectization_stencil = "FD";
  } else if(stencil_type == 2) {
    discrectization_stencil = "FE";
  }
  matrix_type A = Test::generate_structured_matrix2D<matrix_type>(discrectization_stencil,
                                                                  mat_structure);

  mv_type x("X", A.numCols());
  mv_random_read_type t_x(x);
  mv_type y("Y", A.numRows());
  h_mv_type h_x = Kokkos::create_mirror_view(x);
  h_mv_type h_y = Kokkos::create_mirror_view(y);
  h_mv_type h_y_compare = Kokkos::create_mirror(y);

  typename matrix_type::StaticCrsGraphType::HostMirror h_graph = Kokkos::create_mirror(A.graph);
  typename matrix_type::values_type::HostMirror h_values = Kokkos::create_mirror_view(A.values);

  for(int i = 0; i < A.numCols(); i++) {
    h_x(i) = (Scalar) (1.0*(rand()%40)-20.);
  }
  for(int i = 0; i < A.numRows(); i++) {
    h_y(i) = (Scalar) (1.0*(rand()%40)-20.);
  }

  // Error Check Gold Values
  for(int i = 0; i < A.numRows(); i++) {
    int start = h_graph.row_map(i);
    int end = h_graph.row_map(i+1);
    for(int j = start; j < end; j++) {
      h_values(j) = h_graph.entries(j) + i;
    }

    h_y_compare(i) = 0;
    for(int j = start; j < end; j++) {
      Scalar tmp_val = h_graph.entries(j) + i;
      int idx = h_graph.entries(j);
      h_y_compare(i)+=tmp_val*h_x(idx);
    }
  }

  Kokkos::deep_copy(x,h_x);
  Kokkos::deep_copy(y,h_y);
  Kokkos::deep_copy(A.graph.entries,h_graph.entries);
  Kokkos::deep_copy(A.values,h_values);
  typename KokkosSparse::CrsMatrix<Scalar,int,Kokkos::DefaultExecutionSpace,void,int>::values_type x1("X1", A.numCols());
  Kokkos::deep_copy(x1,h_x);
  typename KokkosSparse::CrsMatrix<Scalar,int,Kokkos::DefaultExecutionSpace,void,int>::values_type y1("Y1", A.numRows());

  if(algo == STRUCT) {
    KokkosSparse::Experimental::spmv_struct("N", stencil_type, mat_structure, 1.0, A, x1, 1.0, y1);
  } else if(algo == UNSTR) {
    KokkosSparse::spmv("N", 1.0, A, x1, 1.0, y1);
  }

  // Error Check
  Kokkos::deep_copy(h_y,y1);
  Scalar error = 0;
  Scalar sum = 0;
  for(int i = 0; i < A.numRows(); i++) {
    error += (h_y_compare(i)-h_y(i))*(h_y_compare(i)-h_y(i));
    sum += h_y_compare(i)*h_y_compare(i);
  }

  int num_errors = 0;
  double total_error = 0;
  double total_sum = 0;
  num_errors += (error/(sum==0?1:sum))>1e-5?1:0;
  total_error += error;
  total_sum += sum;

  // Benchmark
  double min_time = 1.0e32;
  double max_time = 0.0;
  double ave_time = 0.0;
  for(int i=0;i<loop;i++) {
    Kokkos::Timer timer;
    // matvec(A,x1,y1,rows_per_thread,team_size,vector_length,test,schedule);
    if(algo == STRUCT) {
      KokkosSparse::Experimental::spmv_struct("N", stencil_type, mat_structure, 1.0, A, x1, 1.0, y1);
    } else if(algo == UNSTR) {
      KokkosSparse::spmv("N", 1.0, A, x1, 1.0, y1);
    }
    Kokkos::fence();
    double time = timer.seconds();
    ave_time += time;
    if(time>max_time) max_time = time;
    if(time<min_time) min_time = time;
  }

  // Performance Output
  double matrix_size = 1.0*((A.nnz()*(sizeof(Scalar)+sizeof(int)) + A.numRows()*sizeof(int)))/1024/1024;
  double vector_size = 2.0*A.numRows()*sizeof(Scalar)/1024/1024;
  double vector_readwrite = (A.nnz()+A.numCols())*sizeof(Scalar)/1024/1024;

  double problem_size = matrix_size+vector_size;
  if(banner) {
    printf("Type NNZ NumRows NumCols ProblemSize(MB) AveBandwidth(GB/s) MinBandwidth(GB/s) MaxBandwidth(GB/s) AveGFlop MinGFlop MaxGFlop aveTime(ms) maxTime(ms) minTime(ms) numErrors\n");
  }
  if(algo == STRUCT) { printf("Struct "); }
  if(algo == UNSTR)  { printf("Unstr  "); }
  printf("%i %i %i %6.2lf ( %6.2lf %6.2lf %6.2lf ) ( %6.3lf %6.3lf %6.3lf ) ( %6.3lf %6.3lf %6.3lf ) %i RESULT\n",A.nnz(), A.numRows(),A.numCols(),problem_size,
          (matrix_size+vector_readwrite)/ave_time*loop/1024, (matrix_size+vector_readwrite)/max_time/1024,(matrix_size+vector_readwrite)/min_time/1024,
         2.0*A.nnz()*loop/ave_time/1e9, 2.0*A.nnz()/max_time/1e9, 2.0*A.nnz()/min_time/1e9,
          ave_time/loop*1000, max_time*1000, min_time*1000,
          num_errors);
  return (int) total_error;
}

void print_help() {
  printf("SPMV_struct benchmark code written by Luc Berger-Vergiat.\n");
  printf("Options:\n");
  printf("  --schedule [SCH]: Set schedule for kk variant (static,dynamic,auto [ default ]).\n");
  printf("  -rpt [K]        : Number of Rows assigned to a thread.\n");
  printf("  -ts [T]         : Number of threads per team.\n");
  printf("  -vl [V]         : Vector-length (i.e. how many Cuda threads are a Kokkos 'thread').\n");
  printf("  -l [LOOP]       : How many spmv to run to aggregate average time. \n");
  printf("  -nx             : How many nodes in x direction. \n");
  printf("  -ny             : How many nodes in y direction. \n");
  printf("  -st             : The stencil type used for discretization: 1 -> FD, 2 -> FE.\n");
}

int main(int argc, char **argv)
{
  // long long int size = 110503; // a prime number
  // int test=KOKKOS;

  int nx = 1000;
  int ny = 1000;
  int stencil_type = 1;
  int rows_per_thread = -1;
  int vector_length = -1;
  int team_size = -1;
  int schedule=AUTO;
  int loop = 100;

  if(argc == 1) {
    print_help();
    return 0;
  }

  for(int i=0;i<argc;i++)
    {
      if((strcmp(argv[i],"-nx" )==0)) {nx=atoi(argv[++i]); continue;}
      if((strcmp(argv[i],"-ny" )==0)) {ny=atoi(argv[++i]); continue;}
      if((strcmp(argv[i],"-st" )==0)) {stencil_type=atoi(argv[++i]); continue;}
      if((strcmp(argv[i],"-rpt")==0)) {rows_per_thread=atoi(argv[++i]); continue;}
      if((strcmp(argv[i],"-ts" )==0)) {team_size=atoi(argv[++i]); continue;}
      if((strcmp(argv[i],"-vl" )==0)) {vector_length=atoi(argv[++i]); continue;}
      if((strcmp(argv[i],"-l"  )==0)) {loop=atoi(argv[++i]); continue;}
      if((strcmp(argv[i],"--schedule")==0)) {
        i++;
        if((strcmp(argv[i],"auto")==0))
          schedule = AUTO;
        if((strcmp(argv[i],"dynamic")==0))
          schedule = DYNAMIC;
        if((strcmp(argv[i],"static")==0))
          schedule = STATIC;
        continue;
      }
      if((strcmp(argv[i],"--help")==0) || (strcmp(argv[i],"-h")==0)) {
        print_help();
        return 0;
      }
    }

  Kokkos::initialize(argc,argv);

  int leftBC = 1, rightBC = 1, bottomBC = 1, topBC = 1;
  Kokkos::Profiling::pushRegion("Structured spmv test");
  int total_errors = test_crs_matrix_singlevec<double>(true, STRUCT,
                                                       nx, ny,
                                                       leftBC, rightBC, bottomBC, topBC,
                                                       stencil_type,
                                                       rows_per_thread,
                                                       team_size,
                                                       vector_length,
                                                       schedule,
                                                       loop);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("Unstructured spmv test");
  test_crs_matrix_singlevec<double>(false, UNSTR,
                                    nx, ny,
                                    leftBC, rightBC, bottomBC, topBC,
                                    stencil_type,
                                    rows_per_thread,
                                    team_size,
                                    vector_length,
                                    schedule,
                                    loop);
  Kokkos::Profiling::popRegion();

  if(total_errors == 0)
    printf("Kokkos::MultiVector Test: Passed\n");
  else
    printf("Kokkos::MultiVector Test: Failed\n");


  Kokkos::finalize();
}
