/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_INL_CUH_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_INL_CUH_

#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {

template<typename OP>
void ElemwiseBinaryOp::RspRspOp(mshadow::Stream<gpu> *s,
                                const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const NDArray &lhs,
                                const NDArray &rhs,
                                const OpReqType req,
                                const NDArray &output,
                                const bool lhs_may_be_dense,
                                const bool rhs_may_be_dense,
                                const bool allow_inplace,
                                const bool scatter) {
  LOG(FATAL) << "GPU not supported for RspRspOp";
}

/*! \brief CSR -op- CSR binary operator for non-canonical NDArray */
template<typename OP>
void ElemwiseBinaryOp::CsrCsrOp(mshadow::Stream<gpu> *s,
                                const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const NDArray &lhs,
                                const NDArray &rhs,
                                const OpReqType req,
                                const NDArray &output) {
  LOG(FATAL) << "GPU not supported for CsrCsrOp";
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_INL_CUH_

