// bin/est-pca.cc

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Copmute Frobenius norm loss on the matrix globally or per-utt: \n"
        " (test-co-act-mat - ref-co-act-mat) \n"
        "The output is a kaldi vector for both options \n"
        "\n"
      "Usage:  compute-fn-loss [options] <ref-coact-mat-rxfilename> <feature-rspecifier> <fn-loss-wspecifier>\n"
        "e.g.:\n"
        "  compute-fn-loss --global=yes ref-coact-mat.ark scp:test.scp ark:fn-loss.ark\n";

    ParseOptions po(usage);

    bool global = false;
    po.Register("global", &global,
		"yes|no, Compute one coactivation matrix on all utterances in the dataset.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ref_mat_rxfilename = po.GetArg(1),
      feature_rspecifier = po.GetArg(2),
      fn_loss_mat_wspecifier = po.GetArg(3);
    
    int32 num_done = 0, num_err = 0;
    int64 count = 0;
    Matrix<double> ref;
    Matrix<double> coact;
    Vector<BaseFloat> loss(1);

    ReadKaldiObject(ref_mat_rxfilename, &ref);
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    BaseFloatVectorWriter fn_loss_writer(fn_loss_mat_wspecifier);
    
    for (; !feat_reader.Done(); feat_reader.Next()) {
      Matrix<double> mat(feat_reader.Value());
      
      if (mat.NumRows() == 0) {
	KALDI_WARN << "Empty feature matrix";
	num_err++;
	continue;
      }
      if (coact.NumCols() == 0 || !global ) {
	coact.Resize(mat.NumCols(), mat.NumCols());
      }

      if (global) {
	coact.AddMatMat(1.0, mat, kTrans, mat, kNoTrans, 1.0);
	count += mat.NumRows();
      } else { 
	coact.AddMatMat(1.0, mat, kTrans, mat, kNoTrans, 0);
	coact.Scale(1.0 / mat.NumRows());
	coact.AddMat(-1.0, ref);
  
	loss(0) = pow(coact.FrobeniusNorm(), 2);      
	fn_loss_writer.Write(feat_reader.Key(), loss);
      }
      num_done++;
    }
    
    if (global) {
      coact.Scale(1.0 / count);
      coact.AddMat(-1.0, ref);

      loss(0) = pow(coact.FrobeniusNorm(), 2);
      fn_loss_writer.Write("all", loss);
    }
    
    KALDI_LOG << "Computed Frobenius Norm loss from " << num_done << " feature files, "
	      << num_err << " with errors.";
    
    if (num_done == 0)
      KALDI_ERR << "No data accumulated.";
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


