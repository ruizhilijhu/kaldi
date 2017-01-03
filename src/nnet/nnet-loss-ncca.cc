// nnet/nnet-loss.cc

// Copyright 2011-2015  Brno University of Technology (author: Karel Vesely)

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

#include <sstream>
#include <iterator>
#include <algorithm>

#include "nnet/nnet-loss-ncca.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet1 {


/* NCCA */ 
void Ncca::Eval(const VectorBase<BaseFloat> &frame_weights,
	       const CuMatrixBase<BaseFloat>& net_out,
	       const CuMatrixBase<BaseFloat>& target,
	       CuMatrix<BaseFloat>* diff) {

  // check inputs, 
  KALDI_ASSERT(net_out.NumRows() == frame_weights.Dim());
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(target.NumCols() == target.NumRows());

  KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(net_out.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(target.Sum()));
   
  int32 num_frames = frame_weights.Sum();
  KALDI_ASSERT(num_frames >= 0.0);

  // for debug 
  static const int32 minibatch_idx = 1;
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(frame_weights, "dbg_frame_weights", false);
    WriteKaldiObject(net_out, "dbg_net_out", false);
    WriteKaldiObject(target, "dbg_target", false);
  }

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;
  
  // compute different matrix (coactivation matrix - target)
  diff_cm_target_.Resize(target.NumRows(), target.NumCols());
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(diff_cm_target_, "dbg_diff_cm_target_0", false);
  }

  diff_cm_target_.AddMatMat(1.0, net_out, kTrans, net_out, kNoTrans, 0);
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(diff_cm_target_, "dbg_diff_cm_target_1", false);
  }

  diff_cm_target_.Scale(1.0 / num_frames);
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(diff_cm_target_, "dbg_diff_cm_target_2", false);
  }

  diff_cm_target_.AddMat(-1.0, target);
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(diff_cm_target_, "dbg_diff_cm_target_3", false);
  }  
  // compute derivative w.r.t. neural nerwork outputs
  *diff = net_out; // initialize
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(*diff, "dbg_diff_0", false);
  }

  diff->AddMatMat(1.0, net_out, kNoTrans, diff_cm_target_, kNoTrans, 0); // (y * diff_cm_target_)  
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(*diff, "dbg_diff_1", false);
  }

  diff->Scale(4.0 / num_frames);
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(*diff, "dbg_diff_2", false);
  }

  diff->MulRowsVec(frame_weights_);
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(*diff, "dbg_diff_3", false);
  }

  // Compute Frobenius norm loss of mini-batch
  double frobenius_norm_loss = pow(diff_cm_target_.FrobeniusNorm(),2);  // sum the matrix,
  
  if ( minibatches_ == minibatch_idx ) {
    KALDI_VLOG(1) << "Frobenius norm loss (of current minibatch): "
		  << frobenius_norm_loss;      
  }

  KALDI_ASSERT(KALDI_ISFINITE(frobenius_norm_loss));
  
  // accumulate
  loss_ += frobenius_norm_loss;
  frames_ += num_frames;
  minibatches_ += 1.0; 
  
  // progressive loss reporting
  {
    static const int32 progress_step = 3600*100;  // 1h
    frames_progress_ += num_frames;
    loss_progress_ += frobenius_norm_loss;
    minibatches_progress_ += 1.0;
    if (frames_progress_ > progress_step) {
      KALDI_VLOG(1) << "ProgressLoss[last "
		    << static_cast<int>(frames_progress_/100/3600) << "h of "
		    << static_cast<int>(frames_/100/3600) << "h]: "
		    << loss_progress_/minibatches_progress_ << " (Ncca)";
      // store
      loss_vec_.push_back(loss_progress_/minibatches_progress_);
      // reset
      frames_progress_ = 0;
      loss_progress_ = 0.0;
      minibatches_progress_ = 0.0;
    }
  }
}

    
 
std::string Ncca::Report() {
  // build the message,
  std::ostringstream oss;
  oss << "AvgLoss: " << loss_/minibatches_ << " (Ncca), "
      << "[minibatches " << minibatches_ << ", frames "
      << frames_ << "]" << std::endl;
  oss << "progress: [";
  std::copy(loss_vec_.begin(), loss_vec_.end(),
	    std::ostream_iterator<float>(oss, " "));
  oss << "]" << std::endl;
  return oss.str();
}
  
}  // namespace nnet1
}  // namespace kaldi
