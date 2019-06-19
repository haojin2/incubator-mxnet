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

/*!
 * \file np_elemwise_binary_op.cc
 * \brief CPU Implementation of numpy elementwise binary function.
 */
#include <mxnet/base.h>
#include "../tensor/elemwise_binary_op.h"

namespace mxnet {
namespace op {

#define MXNET_OPERATOR_REGISTER_NUMPY_BINARY(__name$, __kernel$)         \
NNVM_REGISTER_OP(__name$)                                                               \
.set_num_inputs(2)                                                                      \
.set_num_outputs(1)                                                                     \
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<2, 1>)                       \
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)                           \
.set_attr<nnvm::FInplaceOption>("FInplaceOption",                                       \
  [](const NodeAttrs& attrs){                                                           \
    return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};                           \
  })                                                                                    \
.set_attr<nnvm::FListInputNames>("FListInputNames",                                     \
  [](const NodeAttrs& attrs) {                                                          \
    return std::vector<std::string>{"x1", "x2"};                                        \
  })                                                                                    \
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, __kernel$>)         \
.add_argument("x1", "NDArray-or-Symbol", "The input array.")                            \
.add_argument("x2", "NDArray-or-Symbol", "The input array.")

MXNET_OPERATOR_REGISTER_NUMPY_BINARY(_np_floor_divide, mshadow_op::floor_divide)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

}  // namespace op
}  // namespace mxnet
