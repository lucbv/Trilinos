// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
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
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <numeric>

#ifdef HAVE_MPI
#include "mpi.h"
#endif

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void readRegionInfoFromFile(int& maxRegPerGID,
    int& maxRegPerProc,
    GlobalOrdinal& globalNx,
    GlobalOrdinal& globalNy,
    int& whichCase,
    int& numDimensions,
    std::string regionDataDirectory,
    int myRank)
{
  // Open the file to be read
  FILE *fp;
  std::stringstream fileNameSS;
  fileNameSS << regionDataDirectory << "/myRegionInfo_" << myRank;
  while ((fp = fopen(fileNameSS.str().c_str(),"r") ) == NULL) sleep(1);

  // Read the information
  char command[40];
  int iii = 0;
  fgets(command,80,fp);
  sscanf(command,"%d",&maxRegPerGID);
  while ( command[iii] != ' ') iii++;
  sscanf(&(command[iii+1]),"%d",&maxRegPerProc);
  while ( command[iii] == ' ') iii++;
  while ( command[iii] != ' ') iii++;
  sscanf(&(command[iii+1]),"%d",&globalNx);
  while ( command[iii] == ' ') iii++;
  while ( command[iii] != ' ') iii++;
  sscanf(&(command[iii+1]),"%d",&globalNy);
  while ( command[iii] == ' ') iii++;
  while ( command[iii] != ' ') iii++;
  if      (command[iii+1] == 'M') whichCase = MultipleRegionsPerProc;
  else if (command[iii+1] == 'R') whichCase = RegionsSpanProcs;
  else {fprintf(stderr,"%d: head messed up %s\n",myRank,command); exit(1);}

  if (globalNy == 1)
    numDimensions = 1;
  else
    numDimensions = 2;

  return;
}
