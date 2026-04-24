#include <torch/csrc/distributed/c10d/symm_mem/nccl_ep.hpp>

#ifdef USE_NCCL_EP

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <nccl_ep.h>

namespace c10d::nccl_ep {

// Wraps an at::Tensor as ncclNDTensor_t. Keeps sizes/strides arrays alive
// on the stack as long as this struct is alive.
struct TensorDesc {
    ncclNDTensor_t nd{};
    unsigned int sizes[8]{};
    unsigned int strides[8]{};

    TensorDesc(const at::Tensor& t, ncclEpTensorTag_t tag) {
        TORCH_CHECK(
            t.is_contiguous(),
            "nccl_ep tensors must be memory-contiguous (call .contiguous())");
        nd.version = 1;
        nd.ndim = static_cast<unsigned int>(t.dim());
        for (unsigned int i = 0; i < nd.ndim; i++) {
            sizes[i] = static_cast<unsigned int>(t.size(i));
            // nccl_ep's tensor_is_contiguous() requires strides[i] == 1 for every
            // dimension (same as ncclEpTensorCreate / ep_bench TENSOR_INIT_CONTIG).
            strides[i] = 1;
        }
        nd.sizes = sizes;
        nd.strides = strides;
        nd.datatype = c10d::getNcclDataType(t.scalar_type());
        nd.data = t.data_ptr();
        nd.tag = static_cast<unsigned int>(tag);
        nd.flags = NCCL_EP_TENSOR_FLAG_NONE;
    }
};

#define NCCL_EP_CHECK(expr)                                             \
    do {                                                                \
        ncclResult_t _r = (expr);                                       \
        TORCH_CHECK(_r == ncclSuccess, "nccl_ep error: ", ncclGetErrorString(_r)); \
    } while (0)

static ncclComm_t get_nccl_comm(
    const c10::intrusive_ptr<::c10d::ProcessGroup>& pg) {
    auto* ncclPg = dynamic_cast<c10d::ProcessGroupNCCL*>(
        pg->getBackend(c10::DeviceType::CUDA).get());
    TORCH_CHECK(ncclPg != nullptr, "backend must be a NCCL process group");
    return reinterpret_cast<ncclComm_t>(ncclPg->getCommPtr());
}

NcclEpGroup::~NcclEpGroup() {
    if (group) {
        ncclEpGroupDestroy(
            reinterpret_cast<ncclEpGroup_t>(group),
            cudaStreamDefault);
        group = nullptr;
    }
}

NcclEpHandle::~NcclEpHandle() {
    if (handle) {
        ncclEpHandleDestroy(reinterpret_cast<ncclEpHandle_t>(handle));
        handle = nullptr;
    }
}

c10::intrusive_ptr<NcclEpGroup> nccl_ep_create_group(
    const c10::intrusive_ptr<::c10d::ProcessGroup>& pg,
    int64_t num_experts,
    int64_t max_tokens_per_rank,
    int64_t token_size_bytes,
    int64_t num_qp_per_rank,
    int64_t num_channels) {
    ncclComm_t comm = get_nccl_comm(pg);
    auto stream = at::cuda::getCurrentCUDAStream();

    ncclEpGroupConfig_t config{};
    config.version = 1;
    config.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    config.num_experts = static_cast<unsigned int>(num_experts);
    config.max_tokens_per_rank = static_cast<unsigned int>(max_tokens_per_rank);
    config.token_size_bytes = static_cast<unsigned int>(token_size_bytes);
    config.rdma_buffer_size = NCCL_EP_AUTO;
    config.num_qp_per_rank = static_cast<unsigned int>(num_qp_per_rank);
    config.num_channels = static_cast<unsigned int>(num_channels);

    ncclEpGroup_t ep_group;
    NCCL_EP_CHECK(ncclEpCreateGroup(&ep_group, comm, &config, stream));

    auto result = c10::make_intrusive<NcclEpGroup>();
    result->group = ep_group;
    return result;
}

c10::intrusive_ptr<NcclEpHandle> nccl_ep_create_handle(
    const c10::intrusive_ptr<NcclEpGroup>& group,
    const at::Tensor& topk_idx,
    const std::optional<at::Tensor>& recv_expert_counter) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto ep_group = reinterpret_cast<ncclEpGroup_t>(group->group);

    TensorDesc topk_desc(topk_idx, NCCL_EP_TENSOR_TAG_TOPK_IDX);

    std::optional<TensorDesc> counter_desc;
    ncclNDTensor_t* local_tensors_arr[1]{};
    unsigned int num_local_tensors = 0;
    if (recv_expert_counter) {
        counter_desc.emplace(
            *recv_expert_counter,
            NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE);
        local_tensors_arr[0] = &counter_desc->nd;
        num_local_tensors = 1;
    }

    ncclEpHandle_t ep_handle;
    NCCL_EP_CHECK(ncclEpCreateHandle(
        &ep_handle, ep_group,
        &topk_desc.nd,
        num_local_tensors > 0 ? local_tensors_arr : nullptr,
        num_local_tensors,
        nullptr, stream));

    auto result = c10::make_intrusive<NcclEpHandle>();
    result->handle = ep_handle;
    return result;
}

int64_t nccl_ep_handle_get_num_recv_tokens(
    const c10::intrusive_ptr<NcclEpHandle>& handle) {
    unsigned int num_recv_tokens;
    NCCL_EP_CHECK(ncclEpHandleGetNumRecvTokens(
        reinterpret_cast<ncclEpHandle_t>(handle->handle),
        &num_recv_tokens));
    return static_cast<int64_t>(num_recv_tokens);
}

void nccl_ep_dispatch(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& tokens,
    const at::Tensor& topk_weights,
    const at::Tensor& topk_idx,
    at::Tensor& out_tokens,
    at::Tensor& out_topk_weights,
    at::Tensor& out_topk_idx) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto ep_handle = reinterpret_cast<ncclEpHandle_t>(handle->handle);

    TensorDesc in_tokens(tokens, NCCL_EP_TENSOR_TAG_TOKENS);
    TensorDesc in_weights(topk_weights, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);
    TensorDesc in_idx(topk_idx, NCCL_EP_TENSOR_TAG_TOPK_IDX);
    TensorDesc out_tok(out_tokens, NCCL_EP_TENSOR_TAG_TOKENS);
    TensorDesc out_wts(out_topk_weights, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);
    TensorDesc out_idx_desc(out_topk_idx, NCCL_EP_TENSOR_TAG_TOPK_IDX);

    const ncclNDTensor_t* inputs[] = {&in_tokens.nd, &in_weights.nd, &in_idx.nd};
    ncclNDTensor_t* outputs[] = {&out_tok.nd, &out_wts.nd, &out_idx_desc.nd};
    ncclEpDispatchConfig_t config{};

    NCCL_EP_CHECK(ncclEpDispatch(
        ep_handle,
        inputs, 3,
        outputs, 3,
        nullptr, 0,
        /*send_only=*/0,
        &config, stream));
}

void nccl_ep_combine(
    const c10::intrusive_ptr<NcclEpHandle>& handle,
    const at::Tensor& expert_tokens,
    at::Tensor& out_tokens) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto ep_handle = reinterpret_cast<ncclEpHandle_t>(handle->handle);

    TensorDesc in_tok(expert_tokens, NCCL_EP_TENSOR_TAG_TOKENS);
    TensorDesc out_tok(out_tokens, NCCL_EP_TENSOR_TAG_TOKENS);

    const ncclNDTensor_t* inputs[] = {&in_tok.nd};
    ncclNDTensor_t* outputs[] = {&out_tok.nd};

    NCCL_EP_CHECK(ncclEpCombine(
        ep_handle,
        inputs, 1,
        outputs, 1,
        nullptr, 0,
        /*send_only=*/0,
        nullptr, stream));
}

} // namespace c10d::nccl_ep

#else // USE_NCCL_EP

namespace c10d::nccl_ep {

NcclEpGroup::~NcclEpGroup() = default;
NcclEpHandle::~NcclEpHandle() = default;

static void not_supported() {
    TORCH_CHECK(false, "PyTorch was not built with USE_NCCL_EP=1");
}

c10::intrusive_ptr<NcclEpGroup> nccl_ep_create_group(
    const c10::intrusive_ptr<::c10d::ProcessGroup>&,
    int64_t, int64_t, int64_t, int64_t, int64_t) {
    not_supported();
}

c10::intrusive_ptr<NcclEpHandle> nccl_ep_create_handle(
    const c10::intrusive_ptr<NcclEpGroup>&,
    const at::Tensor&,
    const std::optional<at::Tensor>&) {
    not_supported();
}

int64_t nccl_ep_handle_get_num_recv_tokens(
    const c10::intrusive_ptr<NcclEpHandle>&) {
    not_supported();
}

void nccl_ep_dispatch(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::Tensor&, at::Tensor&, at::Tensor&) {
    not_supported();
}

void nccl_ep_combine(
    const c10::intrusive_ptr<NcclEpHandle>&,
    const at::Tensor&,
    at::Tensor&) {
    not_supported();
}

} // namespace c10d::nccl_ep

#endif // USE_NCCL_EP
