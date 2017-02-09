// nnet/nnet-loss.h

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

#ifndef KALDI_NNET_NNET_LOSS_AANN_NCCA_H_
#define KALDI_NNET_NNET_LOSS_AANN_NCCA_H_

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"
#include "hmm/posterior.h"
#include "nnet/nnet-utils.h"
#include "nnet/nnet-nnet.h"

namespace kaldi {
namespace nnet1 {


class LossItf_aann_ncca {
 public:
  LossItf_aann_ncca() { }
  virtual ~LossItf_aann_ncca() { }

  /// Evaluate cross entropy using target-matrix (supports soft labels),
  virtual void Eval(const VectorBase<BaseFloat> &frame_weights,
		    const CuMatrixBase<BaseFloat> &net_out,
		    Nnet& aann_mlp,
		    CuMatrix<BaseFloat> *diff) = 0;
  
  /// Generate string with error report,
  virtual std::string Report() = 0;
  
  /// Get loss value (frame average),
  virtual BaseFloat FrmAvgLoss() = 0;

  /// Get loss value (minibacth average),
  virtual BaseFloat MbAvgLoss() = 0;
  
};


class AannNcca : public LossItf_aann_ncca {
 public:
  AannNcca():
    frames_(0.0),
    loss_(0.0),
    mse_loss_(0.0),
    minibatches_(0.0),
    frames_progress_(0.0),
    minibatches_progress_(0.0),
    loss_progress_(0.0),
    mse_loss_progress_(0.0)      
  { }
    
  ~AannNcca()
  { }


  /// Evaluate frobenious norm error using target-matrix,
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat>& net_out,
            Nnet& aann_mlp,
            CuMatrix<BaseFloat>* diff);

  /// Generate string with error report
  std::string Report();

  /// Get loss value (minibatch average),
  BaseFloat MbAvgLoss() {
    return loss_ / minibatches_;
  }

  /// Get loss value (frame average),
  BaseFloat FrmAvgLoss() {
    return loss_ / frames_;
  }


  /// Get loss value (frame average), mse
  BaseFloat MseFrmAvgLoss() {
    return mse_loss_ / frames_;
  }

 private:
  double frames_;
  double loss_;
  double mse_loss_;
  double minibatches_;
  double frames_progress_;
  double minibatches_progress_;
  double loss_progress_;
  double mse_loss_progress_;

  std::vector<float> frames_loss_vec_;
  std::vector<float> minibatches_loss_vec_;
  std::vector<float> mse_frames_loss_vec_;

  CuVector<BaseFloat> frame_weights_;
  CuMatrix<BaseFloat> diff_;
  CuMatrix<BaseFloat> diff_pow_2_;
  CuMatrix<BaseFloat> aann_out_;
  CuMatrix<BaseFloat> aann_in_cm_;
  CuMatrix<BaseFloat> aann_out_cm_;
  CuMatrix<BaseFloat> aann_diff_out_in_cm_;
};
 
}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_LOSS_AANN_NCCA_H_

