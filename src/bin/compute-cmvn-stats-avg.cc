// featbin/copy-feats.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage = 
        "Compute the average from first-order, second-order stats and counts with the following format:\n"
        "  [ first-order stats, count; second-order stats, 0 ] \n"
        "   which is used followed by compute-cmvn-stats. \n"
        "Usage: computue-cmvn-stats-avg [options] <cmvn-rspecifier> <avg-wspecifier>\n"
        "e.g.: compute-cmvn-stats-avg ark:- ark,scp:foo.ark,foo.scp\n";
    
    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    int32 num_done = 0;
    // Copying tables of features.
    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);
    
    BaseFloatVectorWriter avg_writer(wspecifier);
    SequentialBaseFloatMatrixReader cmvn_reader(rspecifier);
    for (; !cmvn_reader.Done(); cmvn_reader.Next(), num_done++) {
      Matrix<BaseFloat> cmvn_stats = cmvn_reader.Value();

      int32 dim = cmvn_stats.NumCols() - 1;
      double count = cmvn_stats(0, dim);
      
      Vector<BaseFloat> avg_cnt(dim+1);
      avg_cnt.CopyRowFromMat(cmvn_stats, 0);
      Vector<BaseFloat> avg_stats(dim);
      
      for (int32 d = 0; d < dim; + d++) {
	avg_stats(d) = avg_cnt(d) / count;
      }
     
      avg_writer.Write(cmvn_reader.Key(), avg_stats);
    }
    KALDI_LOG << "Copmuted " << num_done << " average stats.";
    return (num_done != 0 ? 0 : 1);
    
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
