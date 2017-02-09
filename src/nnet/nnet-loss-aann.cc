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

#include "nnet/nnet-loss-aann.h"
#include "nnet/nnet-utils.h"
#include "nnet/nnet-nnet.h"
#include "cudamatrix/cu-math.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet1 {


/* AANN */ 
void Aann::Eval(const VectorBase<BaseFloat> &frame_weights,
	       const CuMatrixBase<BaseFloat>& net_out,
	       Nnet &aann_mlp,
	       CuMatrix<BaseFloat>* diff) {

  // check inputs,
  KALDI_ASSERT(net_out.NumRows() == frame_weights.Dim());
  KALDI_ASSERT(aann_mlp.OutputDim() == net_out.NumCols());
    
  KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(net_out.Sum()));

  int32 num_frames = frame_weights.Sum();
  KALDI_ASSERT(num_frames >= 0.0);

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;

  // for debug 
  static const int32 minibatch_idx = 1;
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(frame_weights, "dbg_frame_weights", false);
    WriteKaldiObject(net_out, "dbg_net_out", false);
  }

  // compute derivative w.r.t. neural nerwork outputs
  aann_mlp.Propagate(net_out, &aann_out_);   // get target from aann
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(aann_out_, "dbg_aann_out", false);
  }
  aann_out_.AddMat(-1.0, net_out);       // y_ae - y
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(aann_out_, "dbg_diff_aann_out_net_out", false);
  }
  aann_mlp.Backpropagate(aann_out_, diff);   // (y_ae-y)*(dy_ae/dy)
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(*diff, "dbg_obj_diff_1", false);
  }
  diff->AddMat(-1.0, aann_out_);     // (y_ae-y)*(dy_ae/dy)-(y_ae-y)
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(*diff, "dbg_obj_diff_2", false);
  }
  diff->MulRowsVec(frame_weights_);  // weighting
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(*diff, "dbg_obj_diff_3", false);
  }
  
  // Compute MeanSquareError loss of mini-batch
  diff_pow_2_ = aann_out_;
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(diff_pow_2_, "dbg_diff_pow_2_1", false);
  }
  diff_pow_2_.MulElements(diff_pow_2_);  // (y - t)^2
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(diff_pow_2_, "dbg_diff_pow_2_2", false);
  }
  diff_pow_2_.MulRowsVec(frame_weights_);  // w*(y - t)^2
  if ( minibatches_ == minibatch_idx ) {
    WriteKaldiObject(diff_pow_2_, "dbg_diff_pow_2_3", false);
  }

  double mean_square_error = 0.5 * diff_pow_2_.Sum();  // sum the matrix,

  KALDI_ASSERT(KALDI_ISFINITE(mean_square_error));
  KALDI_VLOG(1) << "Mean Square Error loss (of current minibatch): "
		  << mean_square_error;      

  // accumulate
  loss_ += mean_square_error;
  frames_ += num_frames;
  minibatches_ += 1.0; 
  
  // progressive loss reporting
  {
    static const int32 progress_step = 3600*100;  // 1h
    frames_progress_ += num_frames;
    loss_progress_ += mean_square_error;
    minibatches_progress_ += 1.0;
    if (frames_progress_ > progress_step) {
      KALDI_VLOG(1) << "ProgressLoss[last "
                    << static_cast<int>(frames_progress_/100/3600) << "h of "
                    << static_cast<int>(frames_/100/3600) << "h]: "
                    << loss_progress_/frames_progress_ << " (Mse)";
      KALDI_VLOG(1) << "MINIBATCH-LEVEL: ProgressLoss[last "
		    << static_cast<int>(frames_progress_/100/3600) << "h of "
		    << static_cast<int>(frames_/100/3600) << "h]: "
		    << loss_progress_/minibatches_progress_ << " (Mse)";
      // store
      frames_loss_vec_.push_back(loss_progress_/frames_progress_);
      minibatches_loss_vec_.push_back(loss_progress_/minibatches_progress_);

      // reset
      frames_progress_ = 0;
      loss_progress_ = 0.0;
      minibatches_progress_ = 0.0;
      
    }
  }    
}

    
std::string Aann::Report() {
  // compute root mean square,
  int32 num_tgt = diff_pow_2_.NumCols();
  BaseFloat root_mean_square = sqrt(loss_/frames_/num_tgt);
  // build the message,
  std::ostringstream oss;
  oss << "AvgLoss: " << loss_/frames_ << " (Mse), "
      << "[RMS " << root_mean_square << ", frames "
      << frames_ << "]" << std::endl;
  oss << "FRAME-LEVEL: progress: [";
  std::copy(frames_loss_vec_.begin(), frames_loss_vec_.end(),
            std::ostream_iterator<float>(oss, " "));
  oss << "]" << std::endl;
  oss << "MINIBATCH-LEVEL: avgLoss: " << loss_/minibatches_ << " (Mse), "
      << std::endl;
  oss << "MINIBATCH-LEVL: progress: [";
  std::copy(minibatches_loss_vec_.begin(), minibatches_loss_vec_.end(),
            std::ostream_iterator<float>(oss, " "));
  oss << "]" << std::endl;
  return oss.str();
}
  
  
}  // namespace nnet1
}  // namespace kaldi

