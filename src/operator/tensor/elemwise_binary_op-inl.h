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
 * \file elemwise_binary_op.h
 * \brief Function definition of elementwise binary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_INL_H_

#include <vector>
#include <algorithm>
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {

/*!
 * \brief Kernel for performing elemwise op between dense and csr matrix
 * \param i            global thread id
 * \param req          type of request
 * \param out          output array
 * \param dns_data     data array of dense input
 * \param csr_data     data array of csr input
 * \param csr_indices  indices array of csr input
 * \param csr_indptr   indptr array of csr input
 * \param num_rows     number of rows of both inputs
 * \param num_cols     number of columns of both inputs
 */
template<int req, typename OP>
struct ElemwiseDnsCsrDnsKernel {
  template<typename DType, typename IType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* out, DType* dns_data,
                                  const DType* csr_data, const IType* csr_indices,
                                  const CType* csr_indptr, const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols) {
    if (i < num_rows) {
      for (int j = csr_indptr[i]; j < csr_indptr[i+1]; ++j) {
        KERNEL_ASSIGN(out[i * num_cols + csr_indices[j]], req,
                      OP::Map(dns_data[i * num_cols + csr_indices[j]], csr_data[j]));
      }
    }
  }
};

/*! \brief DNS -op- CSR binary operator for non-canonical NDArray */
template<typename xpu, typename OP>
void ElemwiseBinaryOp::DnsCsrDnsOp(mshadow::Stream<xpu> *s,
                                   const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const NDArray &dns,
                                   const NDArray &csr,
                                   const OpReqType req,
                                   const NDArray &output,
                                   const bool reverse) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(dns.storage_type(), kDefaultStorage);
  CHECK_EQ(csr.storage_type(), kCSRStorage);
  CHECK(req != kAddTo);
  CHECK(req != kNullOp);
  const bool supported_op = std::is_same<OP, mshadow_op::minus>::value ||
                            std::is_same<OP, mshadow_op::plus>::value;
  CHECK(supported_op == true);
  const nnvm::dim_t num_csr_rows = csr.shape()[0];
  const nnvm::dim_t num_csr_cols = csr.shape()[1];
  TBlob csr_data = csr.data();
  TBlob csr_indices = csr.aux_data(csr::kIdx);
  TBlob csr_indptr = csr.aux_data(csr::kIndPtr);
  MSHADOW_SGL_DBL_TYPE_SWITCH(csr_data.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(csr_indices.type_flag_, IType, {
      MSHADOW_IDX_TYPE_SWITCH(csr_indptr.type_flag_, CType, {
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          if (reverse && std::is_same<OP, mshadow_op::minus>::value) {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::negation, Req>, xpu>::Launch(
              s, output.data().Size(), output.data().dptr<DType>(), dns.data().dptr<DType>());
            if (!csr.storage_initialized()) { return; }
            mxnet_op::Kernel<ElemwiseDnsCsrDnsKernel<Req, mshadow_op::plus>, xpu>::Launch(
              s, num_csr_rows, output.data().dptr<DType>(),
              output.data().dptr<DType>(), csr_data.dptr<DType>(), csr_indices.dptr<IType>(),
              csr_indptr.dptr<CType>(), num_csr_rows, num_csr_cols);
          } else {
            if (req == kWriteTo) {
              mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
                s, output.data().Size(), output.data().dptr<DType>(), dns.data().dptr<DType>());
            }
            if (!csr.storage_initialized()) { return; }
            mxnet_op::Kernel<ElemwiseDnsCsrDnsKernel<Req, OP>, xpu>::Launch(
              s, num_csr_rows, output.data().dptr<DType>(),
              output.data().dptr<DType>(), csr_data.dptr<DType>(), csr_indices.dptr<IType>(),
              csr_indptr.dptr<CType>(), num_csr_rows, num_csr_cols);
          }
        });
      });
    });
  });
}

/*!
 * \brief Kernel for performing elemwise op between dense and rsp tensor
 * \param i            global thread id
 * \param req          type of request
 * \param out          output array
 * \param dns_data     data array of dense input
 * \param rsp_data     data array of rsp input
 * \param rsp_indices  indices array of rsp input
 * \param num_rows     number of rows of both inputs
 * \param nz_rows      number of non-zero rows of rsp tensor
 * \param num_cols     number of columns of both inputs
 */
template<int req, typename OP>
struct ElemwiseDnsRspDnsKernel {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, DType* out, DType* dns_data,
                                  const DType* rsp_data, const IType* rsp_indices,
                                  const nnvm::dim_t num_rows, const nnvm::dim_t nz_rows,
                                  const nnvm::dim_t num_cols) {
    if (i < nz_rows * num_cols) {
      const nnvm::dim_t rsp_idx = i / num_cols;
      const nnvm::dim_t dns_row = rsp_indices[rsp_idx];
      const nnvm::dim_t col = i % num_cols;
      KERNEL_ASSIGN(out[dns_row * num_cols + col], req,
                    OP::Map(dns_data[dns_row * num_cols + col],
                            rsp_data[rsp_idx * num_cols + col]));
    }
  }
};

/*! \brief DNS -op- RSP binary operator for non-canonical NDArray */
template<typename xpu, typename OP>
void ElemwiseBinaryOp::DnsRspDnsOp(mshadow::Stream<xpu> *s,
                                   const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const NDArray &dns,
                                   const NDArray &rsp,
                                   const OpReqType req,
                                   const NDArray &output,
                                   const bool reverse) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(dns.storage_type(), kDefaultStorage);
  CHECK_EQ(rsp.storage_type(), kRowSparseStorage);
  CHECK_EQ(output.data().Size(), dns.data().Size());
  CHECK(req != kAddTo);
  if (req == kNullOp) return;
  const bool supported_op = std::is_same<OP, mshadow_op::minus>::value ||
                            std::is_same<OP, mshadow_op::plus>::value;
  CHECK(supported_op == true) <<
    "Only plus and minus supported now for elemwise operation between default and rsp matrices";
  const nnvm::dim_t num_rows = dns.shape()[0];
  const nnvm::dim_t num_cols = dns.data().Size() / num_rows;
  const nnvm::dim_t nz_rows = rsp.aux_shape(rowsparse::kIdx).Size();
  TBlob rsp_data = rsp.data();
  TBlob rsp_indices = rsp.aux_data(rowsparse::kIdx);

  MSHADOW_SGL_DBL_TYPE_SWITCH(rsp_data.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(rsp_indices.type_flag_, IType, {
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        if (reverse && std::is_same<OP, mshadow_op::minus>::value) {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::negation, Req>, xpu>::Launch(
            s, output.data().Size(), output.data().dptr<DType>(), dns.data().dptr<DType>());
          if (rsp.storage_initialized()) {
            mxnet_op::Kernel<ElemwiseDnsRspDnsKernel<Req, mshadow_op::plus>, xpu>::Launch(
              s, nz_rows * num_cols, output.data().dptr<DType>(),
              output.data().dptr<DType>(), rsp_data.dptr<DType>(), rsp_indices.dptr<IType>(),
              num_rows, nz_rows, num_cols);
          }
        } else {
          if (req == kWriteTo) {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
              s, output.data().Size(), output.data().dptr<DType>(), dns.data().dptr<DType>());
          }
          if (rsp.storage_initialized()) {
            mxnet_op::Kernel<ElemwiseDnsRspDnsKernel<Req, OP>, xpu>::Launch(
              s, nz_rows * num_cols, output.data().dptr<DType>(),
              output.data().dptr<DType>(), rsp_data.dptr<DType>(), rsp_indices.dptr<IType>(),
              num_rows, nz_rows, num_cols);
          }
        }
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_INL_H_
