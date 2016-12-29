// bin/est-pca.cc

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Estimate Co-Activation matrix" 
        "\n"
        "Usage:  est-coact-mat [options] <feature-rspecifier> <coact-mat-wxfilename>\n"
        "e.g.:\n"
        "  est-coact-mat scp:foo.scp some/dir/0.mat\n";

    bool binary = true;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write co-activation matrix in binary mode.");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1),
        coact_mat_wxfilename = po.GetArg(2);

    int32 num_done = 0, num_err = 0;
    int64 count = 0;
    SpMatrix<double> coact_sq;

    SequentialBaseFloatMatrixReader feat_reader(rspecifier);
    
    for (; !feat_reader.Done(); feat_reader.Next()) {
      Matrix<double> mat(feat_reader.Value());
      if (mat.NumRows() == 0) {
	KALDI_WARN << "Empty feature matrix";
	num_err++;
	continue;
      }
      if (coact_sq.NumCols() == 0) {
	coact_sq.Resize(mat.NumCols());
      }
      if (coact_sq.NumCols() != mat.NumCols()) {
	KALDI_WARN << "Feature dimension mismatch " << coact_sq.NumCols() << " vs. " << mat.NumCols();
	num_err++;
	continue;
      }
      coact_sq.AddMat2(1.0, mat, kTrans, 1.0);
      count += mat.NumRows();
      num_done++;
    }
    KALDI_LOG << "Accumulated stats from " << num_done << " feature files, "
	      << num_err << " with errors; " << count << " frames.";
    
    if (num_done == 0)
      KALDI_ERR << "No data accumulated.";
    coact_sq.Scale(1.0 / count);

    WriteKaldiObject(coact_sq, coact_mat_wxfilename, binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


