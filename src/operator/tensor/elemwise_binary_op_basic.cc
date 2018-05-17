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
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op-inl.h"
#include "../nn/mkldnn/mkldnn_ops-inl.h"
#include "../nn/mkldnn/mkldnn_base-inl.h"

namespace mxnet {
namespace op {

/*! \brief binary op handling for the following row sparse inputs/outputs
  rsp, rsp -> rsp,
  dns, rsp -> rsp,
  rsp, dns -> rsp,
  dns, rsp -> dns,
  rsp, dns -> dns,
*/
template<typename OP>
void ElemwiseBinaryOp::RspRspOp(mshadow::Stream<cpu> *s,
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
  using namespace mshadow;
  using namespace mshadow::expr;
  const NDArray& rsp = lhs.storage_type() == kRowSparseStorage ? lhs : rhs;
  const bool is_dense_result = output.storage_type() == kDefaultStorage;
  const bool lhs_is_dense = lhs.storage_type() == kDefaultStorage;
  const bool rhs_is_dense = rhs.storage_type() == kDefaultStorage;
  CHECK(!lhs_is_dense || lhs_may_be_dense) << "rvalue cannot be dense";
  CHECK(!rhs_is_dense || rhs_may_be_dense) << "rvalue cannot be dense";
  CHECK(!lhs_is_dense || !rhs_is_dense);
  MSHADOW_IDX_TYPE_SWITCH(rsp.aux_type(rowsparse::kIdx), IType, {
    MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
      // Only one item at most may be dense (lhs, rhs or result)
      if (rhs_is_dense) {
        // For right-side dense, in order to have sparse output, lhs input zero should
        // always output zero
        CHECK(fabs(static_cast<float>(OP::Map(DType(0), DType(99)))) < 1e-4f);
        CHECK(!is_dense_result);  // Currently not handled
      }
      if (lhs_is_dense) {
        // For left-side dense, in order to have sparse output, lhs input zero should
        // always output zero
        CHECK(fabs(static_cast<float>(OP::Map(DType(99), DType(0)))) < 1e-4f);
        CHECK(!is_dense_result);  // Currently not handled
      }

      // Memory Estimation: This is (roughly) the number of result rows. We may still
      // need to subtract the number of common rows
      bool lhs_in_place = false, rhs_in_place = false;
      const size_t num_rows_l = lhs_is_dense ? lhs.shape()[0] :
                                               lhs.aux_shape(rowsparse::kIdx).Size();
      const size_t num_rows_r = rhs_is_dense ? rhs.shape()[0] :
                                               rhs.aux_shape(rowsparse::kIdx).Size();
      if (is_dense_result) {
        output.CheckAndAlloc();
      } else {
        if (rhs_is_dense || scatter) {
          output.CheckAndAlloc({mshadow::Shape1(num_rows_l)});
        } else if (lhs_is_dense) {
          output.CheckAndAlloc({mshadow::Shape1(num_rows_r)});
        } else {
          lhs_in_place = IsSameArray(lhs, output);
          rhs_in_place = IsSameArray(rhs, output);
          if (!lhs_in_place && !rhs_in_place) {
            output.CheckAndAlloc({mshadow::Shape1(num_rows_l + num_rows_r)});
          } else {
            CHECK_EQ(allow_inplace, true);
            CHECK_EQ(is_dense_result, false);
            if (lhs_in_place) {
              // For in-place, zero L-value must always be zero output
              DCHECK(fabs(static_cast<float>(OP::Map(DType(0), DType(99)))) < DType(1e-3));
            } else {
              // For in-place, zero R-value must always be zero output
              DCHECK(fabs(static_cast<float>(OP::Map(DType(99), DType(0)))) < DType(1e-3));
            }
          }
        }
      }

      // Indices
      const Tensor<cpu, 1, IType> indices_l = lhs_is_dense ?
                                              Tensor<cpu, 1, IType>() :
                                              lhs.aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>(s);
      const Tensor<cpu, 1, IType> indices_r = rhs_is_dense ?
                                              Tensor<cpu, 1, IType>() :
                                              rhs.aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>(s);
      Tensor<cpu, 1, IType> indices_out = is_dense_result ?
                                          Tensor<cpu, 1, IType>() :
                                          output.aux_data(rowsparse::kIdx).FlatTo1D<cpu, IType>(s);

      // Data
      // TODO(cjolivier01): Change to get_with_shape() calls
      const Tensor<cpu, 2, DType> data_l = AsRowise2D<DType>(s, lhs.data());
      const Tensor<cpu, 2, DType> data_r = AsRowise2D<DType>(s, rhs.data());
      Tensor<cpu, 2, DType> out = AsRowise2D<DType>(s, output.data());

      size_t iter_l = 0;
      size_t iter_r = 0;
      size_t iter_out = 0;
      int32_t num_common_rows = 0;

      if (is_dense_result) {
        if (!num_rows_l && !num_rows_r) {
          const size_t all_rows = static_cast<size_t>(lhs.shape()[0]);
          iter_out = FillDense<DType, OP>(s, all_rows, all_rows, req, &out, iter_out);
        }
      }

      while (iter_l < num_rows_l && iter_r < num_rows_r) {
        IType idx_l = lhs_is_dense ? indices_r[iter_r] : indices_l[iter_l];
        IType idx_r = rhs_is_dense ? idx_l : indices_r[iter_r];
        if (lhs_in_place) {
          while (idx_r < idx_l && ++iter_r < num_rows_r) {
            idx_r = indices_r[iter_r];
          }
          if (iter_r >= num_rows_r) {
            break;
          }
        } else if (rhs_in_place) {
          while (idx_l < idx_r && ++iter_l < num_rows_l) {
            idx_l = indices_l[iter_l];
          }
          if (iter_l >= num_rows_l) {
            break;
          }
        }
        if (is_dense_result) {
          iter_out = FillDense<DType, OP>(s, idx_l, idx_r, req, &out, iter_out);
          DCHECK_EQ(iter_out, static_cast<size_t>(std::min(idx_l, idx_r)));
        }
        if (idx_l == idx_r) {
          // Same row
          if (!is_dense_result) {
            indices_out[iter_out] = idx_l;
          }
          Tensor<cpu, 1, DType> lvalue = !lhs_is_dense ? data_l[iter_l++] : data_l[idx_l];
          Tensor<cpu, 1, DType> rvalue = !rhs_is_dense ? data_r[iter_r++] : data_r[idx_r];
          DCHECK_EQ(lvalue.shape_.Size(), rvalue.shape_.Size());
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, cpu>::Launch(
              s, lvalue.shape_.Size(), out[iter_out].dptr_, lvalue.dptr_, rvalue.dptr_);
          });
          num_common_rows++;
        } else if (idx_l < idx_r) {
          // Left only
          if (!is_dense_result) {
            indices_out[iter_out] = idx_l;
          }
          Tensor<cpu, 1, DType> lvalue = !lhs_is_dense ? data_l[iter_l++] : data_l[idx_l];
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            mxnet_op::Kernel<MissingRValueOp<OP, Req>, cpu>::Launch(
              s, lvalue.shape_.Size(), out[iter_out].dptr_, lvalue.dptr_);
          });
        } else {
          // Right only
          if (scatter) {
            ++iter_r;
            continue;  // skip '++iter_out' below
          }
          if (!is_dense_result) {
            indices_out[iter_out] = idx_r;
          }
          Tensor<cpu, 1, DType> rvalue = !rhs_is_dense ? data_r[iter_r++] : data_r[idx_r];
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            mxnet_op::Kernel<MissingLValueOp<OP, Req>, cpu>::Launch(
              s, rvalue.shape_.Size(), out[iter_out].dptr_, rvalue.dptr_);
          });
        }
        ++iter_out;
      }
      // Evaluate the remaining rows beyond the l and r value row intersetion
      while (iter_l < num_rows_l && !lhs_is_dense && !rhs_in_place) {
        if (!is_dense_result) {
          indices_out[iter_out] = indices_l[iter_l];
        } else {
          const IType idx_l = indices_l[iter_l];
          iter_out = FillDense<DType, OP>(s, lhs.shape()[0], idx_l, req, &out, iter_out);
        }
        Tensor<cpu, 1, DType> lvalue = data_l[iter_l++];
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          mxnet_op::Kernel<MissingRValueOp<OP, Req>, cpu>::Launch(
            s, lvalue.shape_.Size(), out[iter_out++].dptr_, lvalue.dptr_);
        });
      }
      while (iter_r < num_rows_r && !rhs_is_dense && !lhs_in_place && !scatter) {
        if (!is_dense_result) {
          indices_out[iter_out] = indices_r[iter_r];
        } else {
          const IType idx_r = indices_r[iter_r];
          iter_out = FillDense<DType, OP>(s, lhs.shape()[0], idx_r, req, &out, iter_out);
        }
        Tensor<cpu, 1, DType> rvalue = data_r[iter_r++];
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          mxnet_op::Kernel<MissingLValueOp<OP, Req>, cpu>::Launch(
            s, rvalue.shape_.Size(), out[iter_out++].dptr_, rvalue.dptr_);
        });
      }
      if (is_dense_result) {
        const size_t all_rows = static_cast<size_t>(lhs.shape()[0]);
        iter_out = FillDense<DType, OP>(s, all_rows, all_rows, req, &out, iter_out);
      } else {
        if (lhs_in_place) {
          CHECK_LE(iter_out, num_rows_l);
        }
        if (rhs_in_place) {
          CHECK_LE(iter_out, num_rows_r);
        }
        DCHECK_LE(iter_out, num_rows_l + num_rows_r);  // Make sure that we didn't overrun
        nnvm::TShape new_shape = output.aux_shape(rowsparse::kIdx);
        CHECK_LE(iter_out, new_shape.Size());
        if (!rhs_is_dense && !lhs_is_dense && !lhs_in_place && !rhs_in_place && !scatter) {
          // Reduce the first-dimension size by the number of common rows
          new_shape[0] -= num_common_rows;
          output.set_aux_shape(rowsparse::kIdx, new_shape);
        }
      }
    });
  });
}

/*! \brief CSR -op- CSR binary operator for non-canonical NDArray */
template<typename OP>
void ElemwiseBinaryOp::CsrCsrOp(mshadow::Stream<cpu> *s,
                                const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const NDArray &lhs,
                                const NDArray &rhs,
                                const OpReqType req,
                                const NDArray &output) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;

  const auto nr_rows = static_cast<size_t>(lhs.shape()[0]);
  if (!nr_rows) {
    return;
  }
  CHECK_EQ(lhs.aux_shape(csr::kIndPtr).Size(), nr_rows + 1);
  const size_t nr_cols = lhs.shape().Size() / nr_rows;

  CHECK_EQ(lhs.shape().Size(), rhs.shape().Size());

  const bool same_lhs_rhs = IsSameArray(lhs, rhs);

  const size_t lhs_nnz = lhs.storage_shape().Size();
  const size_t rhs_nnz = rhs.storage_shape().Size();

  const size_t output_nnz_guess = same_lhs_rhs ? lhs_nnz : lhs_nnz + rhs_nnz;

  output.CheckAndAlloc({mshadow::Shape1(lhs.shape()[0] + 1),
                        mshadow::Shape1(std::min(output_nnz_guess, lhs.shape().Size()))});
  DCHECK_EQ(output.aux_shape(csr::kIndPtr), lhs.aux_shape(csr::kIndPtr));

  MSHADOW_IDX_TYPE_SWITCH(lhs.aux_type(csr::kIdx), IType, {
    MSHADOW_IDX_TYPE_SWITCH(lhs.aux_type(csr::kIndPtr), CType, {
      MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
        const size_t alloc_size = nr_cols * sizeof(IType) + 2 * nr_cols * sizeof(DType);

        Tensor<cpu, 1, uint8_t> workspace =
          ctx.requested[ResourceRequestType::kTempSpace].get_space_typed<cpu, 1, uint8_t>(
            mshadow::Shape1(alloc_size), s);

        // Allocate temp space and partition into three tensors
        mshadow::Tensor<cpu, 1, IType> next(reinterpret_cast<IType *>(workspace.dptr_),
                                            Shape1(nr_cols));
        mshadow::Tensor<cpu, 1, DType> lhs_row(reinterpret_cast<DType *>(
                                                 workspace.dptr_ + nr_cols * sizeof(IType)),
                                               Shape1(nr_cols));
        mshadow::Tensor<cpu, 1, DType> rhs_row;

        OpBase::FillDense<IType>(s, next.shape_.Size(), IType(-1), req, next.dptr_);
        OpBase::FillDense<DType>(s, lhs_row.shape_.Size(), DType(0),  req, lhs_row.dptr_);

        if (!same_lhs_rhs) {
          rhs_row = Tensor<cpu, 1, DType>(lhs_row.dptr_ + nr_cols, Shape1(nr_cols));
          OpBase::FillDense<DType>(s, rhs_row.shape_.Size(), DType(0), req, rhs_row.dptr_);
        } else {
          rhs_row = lhs_row;
        }

        // Column indices
        const Tensor<cpu, 1, IType> col_indices_l = lhs.aux_data(csr::kIdx).FlatTo1D<cpu, IType>(s);
        const Tensor<cpu, 1, IType> col_indices_r = rhs.aux_data(csr::kIdx).FlatTo1D<cpu, IType>(s);
        Tensor<cpu, 1, IType> col_indices_out = output.aux_data(csr::kIdx).FlatTo1D<cpu, IType>(s);

        // Row pointers
        const Tensor<cpu, 1, CType> row_ptr_l = lhs.aux_data(csr::kIndPtr).FlatTo1D<cpu, CType>(s);
        const Tensor<cpu, 1, CType> row_ptr_r = rhs.aux_data(csr::kIndPtr).FlatTo1D<cpu, CType>(s);
        Tensor<cpu, 1, CType> row_ptr_out = output.aux_data(csr::kIndPtr).FlatTo1D<cpu, CType>(s);

        Tensor<cpu, 1, DType>   data_l = lhs.data().FlatTo1D<cpu, DType>(s);
        Tensor<cpu, 1, DType>   data_r = rhs.data().FlatTo1D<cpu, DType>(s);
        Tensor<cpu, 1, DType> data_out = output.data().FlatTo1D<cpu, DType>(s);

        IType nnz = 0;
        row_ptr_out[0] = 0;

        for (IType i = 0; i < static_cast<IType>(nr_rows); i++) {
          IType head = -2;
          IType length = 0;

          // add a row of A to lhs_row
          const IType i_start_l = row_ptr_l[i];
          const IType i_end_l = row_ptr_l[i + 1];
          for (IType jj = i_start_l; jj < i_end_l; jj++) {
            IType col = col_indices_l[jj];
            lhs_row[col] += data_l[jj];

            if (next[col] == -1) {
              next[col] = head;
              head = col;
              ++length;
            }
          }

          if (!same_lhs_rhs) {
            // add a row of B to rhs_row
            const IType i_start_r = row_ptr_r[i];
            const IType i_end_r = row_ptr_r[i + 1];
            for (IType jj = i_start_r; jj < i_end_r; jj++) {
              const IType col = col_indices_r[jj];
              rhs_row[col] += data_r[jj];

              if (next[col] == -1) {
                next[col] = head;
                head = col;
                ++length;
              }
            }
          }

          // scan through columns where A or B has
          // contributed a non-zero entry
          for (IType jj = 0; jj < length; jj++) {
            const DType result = OP::Map(lhs_row[head], rhs_row[head]);

            if (result != 0) {
              col_indices_out[nnz] = head;
              data_out[nnz] = result;
              ++nnz;
            }

            const IType temp = head;
            head = next[head];

            next[temp] = -1;
            lhs_row[temp] = 0;
            if (!same_lhs_rhs) rhs_row[temp] = 0;
          }

          row_ptr_out[i + 1] = nnz;
        }
      });
    });
  });
}

static void ElemwiseAddEx(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
#if MXNET_USE_MKLDNN == 1
  if (SupportMKLDNN(inputs[0]) && SupportMKLDNN(inputs[1])) {
    MKLDNNSumForward(attrs, ctx, inputs, req[0], outputs[0]);
    return;
  } else if (inputs[0].storage_type() == kDefaultStorage
             && inputs[1].storage_type() == kDefaultStorage) {
    // This happens if inputs are supposed to be in MKLDNN format
    // but MKLDNN doesn't support the data type or the shape. We're
    // forced to convert it to the default format.
    std::vector<TBlob> in_blobs(2);
    std::vector<TBlob> out_blobs(1);
    in_blobs[0] = inputs[0].data();
    in_blobs[1] = inputs[1].data();
    out_blobs[0] = outputs[0].data();
    ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::plus>(attrs, ctx, in_blobs,
                                                         req, out_blobs);
    return;
  }
#endif
  ElemwiseBinaryOp::ComputeEx<cpu, op::mshadow_op::plus>(attrs, ctx, inputs,
                                                         req, outputs);
}

static inline bool ElemwiseAddStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int> *in_attrs,
                                          std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  bool ret = ElemwiseBinaryOp::PreferDenseStorageType<true, true, true>(
               attrs, dev_mask, dispatch_mode, in_attrs, out_attrs);
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask
      && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)
      && out_attrs->at(0) == kDefaultStorage) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif
  return ret;
}

MXNET_OPERATOR_REGISTER_BINARY(elemwise_add)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseAddStorageType)
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::plus>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseAddEx)
.set_attr<FResourceRequest>("FResourceRequest",  /* For Sparse CSR */
                            [](const NodeAttrs& attrs) {
                            return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};})
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_add)
.add_alias("_add").add_alias("_plus").add_alias("_Plus")
.describe(R"code(Adds arguments element-wise.

The storage type of ``elemwise_add`` output depends on storage types of inputs

   - elemwise_add(row_sparse, row_sparse) = row_sparse
   - elemwise_add(csr, csr) = csr
   - elemwise_add(default, csr) = default
   - elemwise_add(csr, default) = default
   - elemwise_add(default, rsp) = default
   - elemwise_add(rsp, default) = default
   - otherwise, ``elemwise_add`` generates output with default storage

)code")
.set_attr<nnvm::FGradient>("FGradient", CloneGradient{"_backward_add"});

// specialized gradient add function to do add to optimization
// this must differ from elemwise_add to prevent add to optimization in forward pass.
MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_grad_add, op::mshadow_op::plus);

static void _backward_ElemwiseAddEx(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
#if MXNET_USE_MKLDNN == 1
  if (inputs[0].IsMKLDNNData()) {
    MKLDNNCopy(attrs, ctx, inputs[0], req[0], outputs[0]);
    MKLDNNCopy(attrs, ctx, inputs[0], req[1], outputs[1]);
    return;
  }
#endif
  ElemwiseBinaryOp::BackwardUseNoneEx<cpu, mshadow_op::identity, mshadow_op::identity>(
      attrs, ctx, inputs, req, outputs);
}

static inline bool ElemwiseAddBackwardStorageType(const nnvm::NodeAttrs& attrs,
                                                  const int dev_mask,
                                                  DispatchMode* dispatch_mode,
                                                  std::vector<int> *in_attrs,
                                                  std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 2);
  bool ret = ElemwiseStorageType<1, 2, true, true, true>(attrs, dev_mask, dispatch_mode,
                                                         in_attrs, out_attrs);
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif
  return ret;
}

NNVM_REGISTER_OP(_backward_add)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 0},
                                                                           {0, 1}};
                                })
#if MXNET_USE_MKLDNN == 1
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
#endif
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseNone<
  cpu, mshadow_op::identity, mshadow_op::identity>)
.set_attr<FComputeEx>("FComputeEx<cpu>", _backward_ElemwiseAddEx)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseAddBackwardStorageType);

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_PD(elemwise_sub, op::mshadow_op::minus)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_sub)
.add_alias("_sub").add_alias("_minus").add_alias("_Minus")
.describe(R"code(Subtracts arguments element-wise.

The storage type of ``elemwise_sub`` output depends on storage types of inputs

   - elemwise_sub(row_sparse, row_sparse) = row_sparse
   - elemwise_sub(csr, csr) = csr
   - elemwise_sub(default, csr) = default
   - elemwise_sub(csr, default) = default
   - elemwise_sub(default, rsp) = default
   - elemwise_sub(rsp, default) = default
   - otherwise, ``elemwise_sub`` generates output with default storage

)code")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_sub"});

NNVM_REGISTER_OP(_backward_sub)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 0},
                                                                           {0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseNone<cpu,
  mshadow_op::identity, mshadow_op::negation>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::BackwardUseNoneEx<cpu,
  mshadow_op::identity, mshadow_op::negation>)
.set_attr<FInferStorageType>("FInferStorageType",
                             ElemwiseStorageType<1, 2, true, true, true>);

MXNET_OPERATOR_REGISTER_BINARY(elemwise_mul)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_mul)
.describe(R"code(Multiplies arguments element-wise.

The storage type of ``elemwise_mul`` output depends on storage types of inputs

   - elemwise_mul(default, default) = default
   - elemwise_mul(row_sparse, row_sparse) = row_sparse
   - elemwise_mul(default, row_sparse) = row_sparse
   - elemwise_mul(row_sparse, default) = row_sparse
   - elemwise_mul(csr, csr) = csr
   - otherwise, ``elemwise_mul`` generates output with default storage

)code")
.set_attr<FInferStorageType>("FInferStorageType",
                             ElemwiseBinaryOp::PreferSparseStorageType)
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::mul>)
.set_attr<FComputeEx>("FComputeEx<cpu>",
                      ElemwiseBinaryOp::ComputeDnsLRValueEx<cpu, op::mshadow_op::mul, true, true>)
.set_attr<FResourceRequest>("FResourceRequest",  /* For Sparse CSR */
                              [](const NodeAttrs& attrs) {
                                return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                              })
.add_alias("_mul").add_alias("_Mul")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mul"});

NNVM_REGISTER_OP(_backward_mul)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 1}};
                                })
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseBinaryOp::BackwardUseInStorageType)
.set_attr<FResourceRequest>("FResourceRequest",  /* For Sparse CSR */
                              [](const NodeAttrs& attrs) {
                                return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                              })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseIn<
  cpu, mshadow_op::right, mshadow_op::left>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::BackwardUseInEx<
  cpu, mshadow_op::right, mshadow_op::left>);

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(elemwise_div, op::mshadow_op::div)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_div)
.describe(R"code(Divides arguments element-wise.

The storage type of ``elemwise_div`` output is always dense

)code")
.add_alias("_div").add_alias("_Div")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_div"});

NNVM_REGISTER_OP(_backward_div)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseIn<
  cpu, mshadow_op::div_grad, mshadow_op::div_rgrad>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::BackwardUseInEx<
  cpu, mshadow_op::div_grad, mshadow_op::div_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY(_mod)
.add_alias("_Mod")
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mod"});

NNVM_REGISTER_OP(_backward_mod)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseIn<
  cpu, mshadow_op::mod_grad, mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet
