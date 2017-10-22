#ifndef CAFFE_OPTIMIZATION_DEBUG_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_DEBUG_SOLVER_HPP_

#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver.hpp"

namespace caffe {

// Special Debugging solver 
template <typename Dtype>
class DebugSGDSolver: public SGDSolver<Dtype> {

 protected:
  virtual void ComputeUpdateValue();

  DISABLE_COPY_AND_ASSIGN(DebugSGDSolver);
};


#endif  // CAFFE_OPTIMIZATION_SOLVER_HPP_
