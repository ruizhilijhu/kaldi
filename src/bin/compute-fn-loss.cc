// bin/est-pca.cc

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Copmute Frobenius norm loss on the matrix: \n"
        " test-co-act-mat - ref-co-act-mat "
        "\n"
      "Usage:  compute-fn-loss [options] <ref-coact-mat-rxfilename> <feature-rspecifier> <fn-loss-wspecifier>\n"
        "e.g.:\n"
        "  compute-fn-loss ref-coact-mat.ark scp:test.scp ark:fn-loss.ark\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ref_mat_rxfilename = po.GetArg(1),
      feature_rspecifier = po.GetArg(2),
      fn_loss_mat_wspecifier = po.GetArg(3);
    
    int32 num_done = 0, num_err = 0;
    SpMatrix<double> ref_sq;

    bool binary_in;
    Input ki(ref_mat_rxfilename, &binary_in);
    ref_sq.Read(ki.Stream(), binary_in);
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    BaseFloatVectorWriter fn_loss_writer(fn_loss_mat_wspecifier);
    
    for (; !feat_reader.Done(); feat_reader.Next()) {
      Matrix<double> mat(feat_reader.Value());
      SpMatrix<double> coact_sq;
      Vector<BaseFloat> loss(1);
      
      if (mat.NumRows() == 0) {
	KALDI_WARN << "Empty feature matrix";
	num_err++;
	continue;
      }
      if (coact_sq.NumCols() == 0) {
	coact_sq.Resize(mat.NumCols());
      }
      coact_sq.AddMat2(1.0, mat, kTrans, 1.0);
      coact_sq.Scale(1.0 / mat.NumRows());
      coact_sq.AddSp(-1.0, ref_sq);
      
      loss(0) = pow(coact_sq.FrobeniusNorm(), 2);      
      fn_loss_writer.Write(feat_reader.Key(), loss);      
      num_done++;
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


