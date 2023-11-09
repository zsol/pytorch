#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/UCCForNCCL.hpp>
#include <mutex>
#include <sstream>

#ifdef USE_C10D_NCCL

#include <exception>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/profiler/combined_traceback.h>

#include <torch/csrc/cuda/nccl.h>
#include <torch/torch.h>

namespace c10d {

constexpr const char* const kNCCLAbortedCommStoreKey = "NCCLABORTEDCOMM";

namespace {

#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || (NCCL_MAJOR == 2) && (NCCL_MINOR >= 10))
#define NCCL_HAS_AVG 1
#endif

// NCCL op mapping
const std::map<ReduceOp::RedOpType, ncclRedOp_t> ncclOp = {
    {ReduceOp::MIN, ncclMin},
    {ReduceOp::MAX, ncclMax},
    {ReduceOp::SUM, ncclSum},
    {ReduceOp::PRODUCT, ncclProd},
#ifdef NCCL_HAS_AVG
    {ReduceOp::AVG, ncclAvg},
#endif
};

// NCCL type typing
std::map<at::ScalarType, ncclDataType_t> ncclDataType = {
    {at::kChar, ncclInt8},
    {at::kByte, ncclUint8},
    {at::kFloat, ncclFloat},
    {at::kDouble, ncclDouble},
    {at::kInt, ncclInt32},
    {at::kLong, ncclInt64},
    {at::kHalf, ncclHalf},
    {at::kBool, ncclUint8},
#if HAS_NCCL_BF16_DATATYPE
    {at::kBFloat16, ncclBfloat16},
#endif
};

// Helper function that gets the data type and issues error if not supported
ncclDataType_t getNcclDataType(at::ScalarType type) {
  auto it = ncclDataType.find(type);
  TORCH_CHECK_WITH(
      TypeError,
      it != ncclDataType.end(),
      "Input tensor data type is not supported for NCCL process group: ",
      type);
  return it->second;
}

#ifdef ENABLE_NCCL_PREMUL_SUM_SUPPORT
template <typename T, ncclDataType_t dataType>
ncclRedOpRAII unpackPreMulSum(
    const ReduceOp& reduceOp,
    const ncclComm_t& comm,
    int dev_in_group) {
  const auto* preMulSupplement =
      reinterpret_cast<NCCLPreMulSumSupplement*>(reduceOp.supplement_.get());
  ncclRedOp_t preMulSum;
  bool has_tensor = preMulSupplement->tensor_factor.defined();
  auto residence = has_tensor ? ncclScalarDevice : ncclScalarHostImmediate;
  const T* ptr_factor = has_tensor
      ? preMulSupplement->tensor_factor.const_data_ptr<T>()
      : nullptr;
  T scalar_factor = T(preMulSupplement->double_factor);
  ncclRedOpCreatePreMulSum(
      &preMulSum,
      // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html#ncclredopcreatepremulsum
      // tells us that the scalar input is strictly a multiplier.
      /*scalar=*/has_tensor ? const_cast<T*>(ptr_factor) : &scalar_factor,
      dataType,
      residence,
      comm);
  return ncclRedOpRAII(preMulSum, comm);
}
#endif

ncclRedOpRAII getNcclReduceOp(
    const ReduceOp& reduceOp,
    at::Tensor& input,
    const ncclDataType_t& dataType,
    const ncclComm_t& comm,
    int dev_in_group) {
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
        // For bool tensors, map sum to max, which both represent a bitwise or.
        // This is to prevent overflow issues with sum, since we use uint8 to
        // represent a bool (see ncclDataType mapping).
        return ncclMax;
      }
#ifdef NCCL_HAS_AVG
      if (reduceOp == ReduceOp::AVG) {
        C10_THROW_ERROR(
            TypeError, "Cannot use ReduceOp.AVG with boolean inputs");
      }
#endif
    }
    if (reduceOp == ReduceOp::PREMUL_SUM) {
#ifdef ENABLE_NCCL_PREMUL_SUM_SUPPORT
      switch (dataType) {
        case ncclHalf:
          return unpackPreMulSum<at::Half, ncclHalf>(
              reduceOp, comm, dev_in_group);
        case ncclFloat:
          return unpackPreMulSum<float, ncclFloat>(
              reduceOp, comm, dev_in_group);
        case ncclDouble:
          return unpackPreMulSum<double, ncclDouble>(
              reduceOp, comm, dev_in_group);
        default:
          C10_THROW_ERROR(
              TypeError, "PreMulSum Data type must be half, float, or double");
          ncclRedOp_t unused;
          return unused;
      }
#else
      C10_THROW_ERROR(ValueError, "PreMulSum requires NCCL>=2.11.1");
#endif
    }
    return ncclOp.at(reduceOp);
  } catch (const std::out_of_range& e) {
    switch (reduceOp) {
      case ReduceOp::AVG:
        C10_THROW_ERROR(
            ValueError,
            c10::str(
                "AVG requires NCCL 2.10+. The current version is ",
                NCCL_MAJOR,
                ".",
                NCCL_MINOR));
        break;
      case ReduceOp::BAND:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BAND with NCCL");
        break;
      case ReduceOp::BOR:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BOR with NCCL");
        break;
      case ReduceOp::BXOR:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BXOR with NCCL");
        break;
      default:
        C10_THROW_ERROR(ValueError, "Unhandled ReduceOp");
        break;
    }
  }
}

// Get the deviceList String from the list of devices
std::string getKeyFromDevices(const std::vector<at::Device>& devices) {
  std::string deviceList;
  for (auto& device : devices) {
    if (deviceList.empty()) {
      deviceList = std::to_string(device.index());
    } else {
      deviceList += "," + std::to_string(device.index());
    }
  }
  return deviceList;
}

std::string getKeySendRecv(int myRank, int peer) {
  int lowRank = myRank < peer ? myRank : peer;
  int highRank = myRank < peer ? peer : myRank;
  std::string sendRecvPair =
      std::to_string(lowRank) + ":" + std::to_string(highRank);
  return sendRecvPair;
}

// Get the list of devices from list of tensors
std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    // tensors must all be on the same device, or all on distinct devices.
    // The line below assumes that constraint has already been enforced
    // (by check_gpu_tensors_same_device or
    // check_gpu_tensors_different_devices).
    if (res.size() == 0 || tensor.device() != res[0]) {
      res.push_back(tensor.device());
    }
  }
  return res;
}

// Return CUDA device with ordinal given by input rank.
at::Device getDeviceForRank(int rank) {
  TORCH_CHECK_WITH(ValueError, rank >= 0, "Invalid rank ", rank);
  auto numGPUs = at::cuda::getNumGPUs();
  int16_t deviceIdx = static_cast<int16_t>(rank % numGPUs);
  return at::Device(at::DeviceType::CUDA, deviceIdx);
}

// [Sync Streams] Helper that lets the input ncclStreams to wait for the current
// stream. NCCL communications run on ncclStreams, but input tensors are
// allocated on different streams (i.e., current streams). Communications on
// ncclStreams cannot start before pending input tensor ops on current streams
// finish. Otherwise, ops on two streams might read/write same tensors
// concurrently.
//
// The synchronization above alone is not enough. We also need to make sure
// input tensors are not freed before their usages on ncclStreams finish. This
// can be achieved by calling c10::cuda::CUDACachingAllocator::recordStream,
// which remembers the usage stream (ncclStream), creates an event on the usage
// stream when GC attempts to free the input tensor, and delays GC until that
// event is done.
void syncStreams(
    const std::vector<at::Device>& devices,
    std::vector<at::cuda::CUDAEvent>& ncclEvents,
    std::vector<at::cuda::CUDAStream>& ncclStreams) {
  for (const auto i : c10::irange(devices.size())) {
    at::cuda::CUDAStream& ncclStream = ncclStreams[i];
    at::cuda::CUDAEvent& ncclEvent = ncclEvents[i];
    ncclEvent.record(at::cuda::getCurrentCUDAStream(devices[i].index()));
    ncclEvent.block(ncclStream);
  }
}

// Given a ncclUniqueId, convert it to a string representation that can be put
// in the store.
std::string buildNcclUniqueIdStr(const ncclUniqueId& ncclID) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&ncclID);
  std::ostringstream oss;
  for (const auto i : c10::irange(NCCL_UNIQUE_ID_BYTES)) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

std::string getNcclAbortedCommStoreKey(const std::string ncclIdStr) {
  return std::string(kNCCLAbortedCommStoreKey) + ":" + ncclIdStr;
}

// Returns exception's what() given an exception_ptr instance.
std::string getExceptionMsgFromExceptionPtr(
    const std::exception_ptr& exceptionPtr) {
  TORCH_CHECK(exceptionPtr != nullptr);
  try {
    std::rethrow_exception(exceptionPtr);
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "Unknown exception type";
  }
}

inline void errorIfCapturingNonCapturableNCCL(c10::cuda::CaptureStatus status) {
  // parentheses avoid some compiler warnings
  static const uint64_t min_version =
      (((uint64_t)2) << 32) + (((uint64_t)9) << 16) + ((uint64_t)6);
  static const uint64_t cur_version = torch::cuda::nccl::version();
  if (cur_version < min_version) {
    TORCH_CHECK_WITH(
        NotImplementedError,
        status == c10::cuda::CaptureStatus::None,
        "Capturing NCCL collectives is only allowed with NCCL >= 2.9.6");
  }
}

} // namespace

namespace {
std::string pickle_str(const c10::IValue& v) {
  std::vector<char> result;
  {
    auto writer = [&](const char* data, size_t size) {
      result.insert(result.end(), data, data + size);
    };
    torch::jit::Pickler pickler(
        writer, nullptr, nullptr, nullptr, nullptr, false);
    pickler.protocol();
    pickler.pushIValue(v);
    pickler.stop();
  }
  return std::string(result.begin(), result.end());
}
c10::Dict<c10::IValue, c10::IValue> new_dict() {
  return c10::Dict<c10::IValue, c10::IValue>(
      c10::AnyType::get(), c10::AnyType::get());
}
c10::List<c10::IValue> new_list() {
  return c10::List<c10::IValue>(c10::AnyType::get());
}

} // namespace

// Map from each communicator to its device index.
// This map is used when register/deregister cache segments from cache
// allocator. See design notes below:
// - Each segment should be registered only to the communicator on the
//   same device.
// - We cannot reuse devNCCLCommMap_ in each ProcessGroup because the key may be
//   ranks rather than device in point-to-point case.
// - This map has also to be maintained as global variable since the register
//   hooks are called outside the scope of any PG, thus we need traverse
//   communicators in all PGs.
static std::unordered_map<std::shared_ptr<NCCLComm>, int> ncclCommDevIdxMap;
static std::mutex ncclCommDevIdxMapMutex;
static bool allocatorHooksAttached = false;
void cacheAllocatorRegisterHook(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  // Register after SEGMENT_ALLOC
  if (te.action_ !=
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
    return;
  }

  std::lock_guard<std::mutex> lock(ncclCommDevIdxMapMutex);
  for (auto& it : ncclCommDevIdxMap) {
    auto& ncclComm = it.first;
    auto& devIdx = it.second;
    if (te.device_ == devIdx) {
      ncclComm->registerSegment(reinterpret_cast<void*>(te.addr_), te.size_);
    }
  }
}

void cacheAllocatorDeregisterHook(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  // deregister before SEGMENT_FREE
  if (te.action_ !=
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE) {
    return;
  }

  std::lock_guard<std::mutex> lock(ncclCommDevIdxMapMutex);
  for (auto& it : ncclCommDevIdxMap) {
    auto& ncclComm = it.first;
    auto& devIdx = it.second;
    if (te.device_ == devIdx) {
      ncclComm->deregisterSegment(reinterpret_cast<void*>(te.addr_));
    }
  }
}

struct NCCLTraceBuffer {
  static NCCLTraceBuffer* get() {
    // intentionally leak on exit
    // because this will hold python state that may get destructed
    static NCCLTraceBuffer* instance = new NCCLTraceBuffer();
    return instance;
  }
  NCCLTraceBuffer() {
    max_entries_ = parseEnvVarIntDefault("TORCH_NCCL_TRACE_BUFFER_SIZE", 0);
    enabled_ = max_entries_ > 0;
  }
  using EventList = std::vector<at::cuda::CUDAEvent>;
  struct Entry {
    size_t id_; // incremented id in the trace buffer
                // used to figure out where in the circular entries
                // buffer this entry will be located to
                // update state information
    size_t pg_id_;
    size_t seq_id_; // as tracked by the process group
    const char* profiling_name_;

    std::shared_ptr<torch::CapturedTraceback> traceback_;
    // we borrow pointser to start_ and end_ so we can query the state
    // on reporting. However, once the event is completed, the call
    // to `complete` will clear these.
    EventList *start_, *end_;
    const char* state_ = "scheduled";

    // size information for input/output tensors
    c10::SmallVector<int, 4> input_dims_;
    c10::SmallVector<int, 4> output_dims_;
    c10::SmallVector<int64_t, 8> sizes_; // flattened from inputs, outputs
  };

  bool enabled_ = false;
  std::mutex mutex_;
  std::vector<Entry> entries_;
  size_t max_entries_ = 0;
  size_t next_ = 0;
  size_t id_ = 0;

  c10::optional<size_t> record(
      size_t pg_id,
      size_t seq_id,
      const char* profiling_name,
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs,
      EventList* start,
      EventList* end) {
    if (!enabled_) {
      return c10::nullopt;
    }
    auto traceback = torch::CapturedTraceback::gather(true, true, true);
    std::lock_guard<std::mutex> guard(mutex_);

    auto te = Entry{
        id_,
        pg_id,
        seq_id,
        profiling_name,
        std::move(traceback),
        std::move(start),
        std::move(end)};

    for (const auto& input : inputs) {
      c10::IntArrayRef sizes = input.sizes();
      te.input_dims_.push_back(sizes.size());
      te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
    }

    for (const auto& output : outputs) {
      c10::IntArrayRef sizes = output.sizes();
      te.output_dims_.push_back(sizes.size());
      te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
    }

    if (entries_.size() < max_entries_) {
      entries_.emplace_back(std::move(te));
    } else {
      entries_[next_++] = std::move(te);
      if (next_ == max_entries_) {
        next_ = 0;
      }
    }
    return id_++;
  }

  std::vector<Entry> dump_entries() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<Entry> result;
    result.reserve(entries_.size());
    result.insert(result.end(), entries_.begin() + next_, entries_.end());
    result.insert(result.end(), entries_.begin(), entries_.begin() + next_);
    // query any remaining events
    for (auto& r : result) {
      if (r.start_ != nullptr) {
        bool started = true;
        for (auto& ev : *r.start_) {
          if (!ev.query()) {
            started = false;
            break;
          }
        }
        if (started) {
          r.state_ = "started";
        }
        r.start_ = nullptr;
      }
      if (r.end_ != nullptr) {
        bool completed = true;
        for (auto& ev : *r.end_) {
          if (!ev.query()) {
            completed = false;
            break;
          }
        }
        if (completed) {
          r.state_ = "completed";
        }
        r.end_ = nullptr;
      }
    }
    return result;
  }

  void complete(c10::optional<size_t> id) {
    if (!enabled_ || !id) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    auto& entry = entries_.at(*id % max_entries_);
    if (entry.id_ == *id) {
      entry.state_ = "completed";
      entry.start_ = entry.end_ = nullptr;
    }
  }

  std::string dump() {
    auto result = dump_entries();
    auto entries = new_list();
    c10::IValue pg_id_s = "pg_id";
    c10::IValue seq_id_s = "seq_id";
    c10::IValue profiling_name_s = "profiling_name";
    c10::IValue input_sizes_s = "input_sizes";
    c10::IValue output_sizes_s = "output_sizes";

    c10::IValue frames_s = "frames";
    c10::IValue state_s = "state";
    c10::IValue line_s = "line";
    c10::IValue name_s = "name";
    c10::IValue filename_s = "filename";

    std::vector<torch::CapturedTraceback*> tracebacks;
    for (auto& e : result) {
      tracebacks.push_back(e.traceback_.get());
    }
    torch::SymbolizedTracebacks stracebacks = torch::symbolize(tracebacks);
    std::vector<c10::IValue> all_frames;
    for (const auto& f : stracebacks.all_frames) {
      auto d = new_dict();
      d.insert(name_s, f.funcname);
      d.insert(filename_s, f.filename);
      d.insert(line_s, int64_t(f.lineno));
      all_frames.emplace_back(std::move(d));
    }

    for (auto i : c10::irange(result.size())) {
      auto& e = result.at(i);
      auto& tb = stracebacks.tracebacks.at(i);
      auto dict = new_dict();
      dict.insert(pg_id_s, int64_t(e.pg_id_));
      dict.insert(seq_id_s, int64_t(e.seq_id_));
      dict.insert(profiling_name_s, e.profiling_name_);

      auto it = e.sizes_.begin();
      auto read_sizes = [&](const c10::SmallVector<int, 4>& dims) {
        auto sizes = new_list();
        for (auto dim : dims) {
          auto arg_sizes = new_list();
          for (auto i : c10::irange(dim)) {
            (void)i;
            arg_sizes.push_back(*it++);
          }
          sizes.push_back(arg_sizes);
        }
        return sizes;
      };

      dict.insert(input_sizes_s, read_sizes(e.input_dims_));
      dict.insert(output_sizes_s, read_sizes(e.output_dims_));
      dict.insert(state_s, e.state_);
      auto frames = new_list();
      for (int64_t frame : tb) {
        frames.push_back(all_frames.at(frame));
      }
      dict.insert(frames_s, frames);
      entries.push_back(dict);
    }
    return pickle_str(entries);
  }
};

std::string dump_nccl_trace() {
  return NCCLTraceBuffer::get()->dump();
}

const int64_t ProcessGroupNCCL::kWatchdogThreadSleepMillis = 1000;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;
thread_local uint64_t ProcessGroupNCCL::ncclActiveGroupCounter_ = 0;

std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupNCCL::WorkNCCL& workNCCL) {
  std::string workInfo;
  workInfo = c10::str(
      "WorkNCCL(",
      "SeqNum=",
      workNCCL.seq_,
      ", OpType=",
      opTypeToString(workNCCL.opType_),
      ", NumelIn=",
      workNCCL.numelIn_,
      ", NumelOut=",
      workNCCL.numelOut_,
      ", Timeout(ms)=",
      workNCCL.opTimeout_.count(),
      ")");
  return output << workInfo;
}

ProcessGroupNCCL::WorkNCCL::WorkNCCL(
    const std::vector<at::Device>& devices,
    int rank,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputs,
    bool desyncDebug,
    bool enableTiming)
    : Work(rank, opType, profilingTitle, inputs),
      devices_(devices),
      workStartTime_(std::chrono::steady_clock::now()),
      seq_(seq),
      timingEnabled_(enableTiming) {
  // Creates the CUDA event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = cudaEventDisableTiming.
  if (enableTiming) {
    ncclStartEvents_ = std::make_shared<std::vector<at::cuda::CUDAEvent>>();
    ncclStartEvents_->reserve(devices.size());
    for (uint32_t i = 0; i < devices.size(); ++i) {
      ncclStartEvents_->emplace_back(at::cuda::CUDAEvent(cudaEventDefault));
    }
  }
  ncclEndEvents_ = std::make_shared<std::vector<at::cuda::CUDAEvent>>();
  ncclEndEvents_->reserve(devices.size());
  for (uint32_t i = 0; i < devices.size(); ++i) {
    ncclEndEvents_->emplace_back(at::cuda::CUDAEvent(
        enableTiming ? cudaEventDefault : cudaEventDisableTiming));
  }
  ncclComms_.resize(devices.size());
}

ProcessGroupNCCL::WorkNCCL::WorkNCCL(const WorkNCCL& w)
    : Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkNCCL>(w),
      devices_(w.devices_),
      ncclStartEvents_(w.ncclStartEvents_),
      ncclEndEvents_(w.ncclEndEvents_),
      ncclComms_(w.ncclComms_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_),
      startTraceUpdated_(w.startTraceUpdated_),
      numelIn_(w.numelIn_),
      numelOut_(w.numelOut_),
      store_(w.store_),
      timingEnabled_(w.timingEnabled_),
      trace_id_(w.trace_id_) {
  exception_ = w.exception_;
}

ProcessGroupNCCL::WorkNCCL::~WorkNCCL() = default;

bool ProcessGroupNCCL::WorkNCCL::isCompleted() {
  checkAndSetException();
  return exception() || finishedGPUExecutionInternal();
}

bool ProcessGroupNCCL::WorkNCCL::isStarted() {
  checkAndSetException();
  return exception() || startedGPUExecutionInternal();
}

bool ProcessGroupNCCL::WorkNCCL::isSuccess() const {
  if (exception()) {
    // Already detected an exception.
    return false;
  }

  return !checkForNCCLErrors(ncclComms_) && finishedGPUExecutionInternal();
}

void ProcessGroupNCCL::WorkNCCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }

  auto exception_ptr = checkForNCCLErrors(ncclComms_);
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
  if (exception_) {
    LOG(INFO) << "[Rank " << rank_ << "]"
              << " found async exception when checking for NCCL errors: "
              << getExceptionMsgFromExceptionPtr(exception_);
  }
}

void ProcessGroupNCCL::WorkNCCL::setException(
    std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
}

// Helper that checks if the NCCL kernels are completed on the GPUs
bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecution() {
  checkAndSetException();
  return finishedGPUExecutionInternal();
}

bool ProcessGroupNCCL::WorkNCCL::startedGPUExecutionInternal() const {
  // if timing is disabled we won't have allocated start events
  if (!timingEnabled_) {
    return false;
  }
  for (const auto i : c10::irange(devices_.size())) {
    // Checking the work's corresponding CUDA events' status
    if (!(*ncclStartEvents_)[i].query()) {
      return false;
    }
  }

  return true;
}

bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const {
  for (const auto i : c10::irange(devices_.size())) {
    // Checking the work's corresponding CUDA events' status
    if (!(*ncclEndEvents_)[i].query()) {
      return false;
    }
  }

  return true;
}

bool ProcessGroupNCCL::WorkNCCL::checkTimeout(
    c10::optional<std::chrono::milliseconds> timeout) {
  auto currentTimepoint = std::chrono::steady_clock::now();
  auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTimepoint - workStartTime_);
  auto workTimeout = timeout ? *timeout : opTimeout_;

  if (timeElapsed < workTimeout)
    return false;

  // Timed out

  // There is already an error, we don't override it
  if (exception())
    return true;

  std::string exceptionMsg = c10::str(
      "[Rank ",
      rank_,
      "] ",
      "Watchdog caught collective operation timeout: ",
      *this,
      " ran for ",
      timeElapsed.count(),
      " milliseconds before timing out.");

  LOG(ERROR) << exceptionMsg;
  std::exception_ptr exception_ptr =
      std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exceptionMsg));
  setException(exception_ptr);
  return true;
}

void ProcessGroupNCCL::WorkNCCL::handleException(
    ErrorHandlingMode errorHandling) {
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some NCCL operations have failed or timed out. Due to the ",
        "asynchronous nature of CUDA kernels, subsequent GPU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupNCCL.WorkNCCL.handleException");

    if (SHOULD_TEAR_DOWN(errorHandling)) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      LOG(ERROR) << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

void ProcessGroupNCCL::WorkNCCL::synchronize() {
  // Call Synchronize without a timeout. We use this method to avoid adding a
  // timeout argument to the public synchronize API.
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupNCCL::WorkNCCL::synchronizeStreams() {
  for (const auto i : c10::irange(devices_.size())) {
    auto currentStream = at::cuda::getCurrentCUDAStream(devices_[i].index());
    // Block the current stream on the NCCL stream
    (*ncclEndEvents_)[i].block(currentStream);
  }

  if (avoidRecordStreams_) {
    stashed_for_allocator_safety_->clear();
  }
}

// Waiting on the work's corresponding CUDA events
void ProcessGroupNCCL::WorkNCCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  synchronizeStreams();

  // In case of blocking, wait for the operation to complete.
  if (blockingWait_) {
    while (!isCompleted()) {
      bool timedOut = checkTimeout(
          timeout == kNoTimeout ? c10::nullopt : c10::make_optional(timeout));
      // Explicitly abort ncclComms here before throwing this timed out
      // exception to users.
      // If throwing timed out excepiton without aborting nccl communicators
      // here, it was observed that CUDA GPU will have 100% utilization and
      // can not run new events successfully.
      if (timedOut) {
        std::string exceptionMsg = c10::str(
            "[Rank ",
            rank_,
            "] Work ",
            (*this),
            " timed out in blocking wait (NCCL_BLOCKING_WAIT=1).");
        LOG(ERROR) << exceptionMsg;
        break;
      }
      // Yield
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
    // exception() includes timeout and error during blocking wait
    if (exception()) {
      // Abort NCCL communicators
      abort();
      // Throw exception (from main thread here)
      handleException(TearDown);
    }
  }

  // Device synchronize only after we've completed timeout checks.
  if (!barrierTensors_.empty()) {
    // If we use the work to do barrier, we should block here
    at::cuda::OptionalCUDAGuard gpuGuard;
    for (auto& device : devices_) {
      gpuGuard.set_index(device.index());
      AT_CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
}

// Same as calling synchronize().
bool ProcessGroupNCCL::WorkNCCL::wait(std::chrono::milliseconds timeout) {
  RECORD_PARAM_COMMS(
      static_cast<int>(this->seq_), // seq
      0, // process group ptr
      rank_, // rank
      "wait", // colName
      0, // inSize
      0, // outSize
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      static_cast<int>(devices_.size())); // worldSize
  synchronizeInternal(timeout);
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupNCCL::WorkNCCL::abort() {
  // Abort all communicators of this work
  for (const auto& ncclComm : ncclComms_) {
    ncclComm->ncclCommAbort();
  }

  ncclCommDevIdxMapMutex.lock();
  for (const auto& comm : ncclComms_) {
    ncclCommDevIdxMap.erase(comm);
  }
  ncclCommDevIdxMapMutex.unlock();
}

ProcessGroupNCCL::CoalescedWorkNCCL::CoalescedWorkNCCL(
    std::vector<ProcessGroupNCCL::WorkNCCL> works,
    int rank,
    OpType opType)
    : Work(rank, opType, nullptr), works_(std::move(works)) {}

ProcessGroupNCCL::CoalescedWorkNCCL::~CoalescedWorkNCCL() = default;

c10::intrusive_ptr<ProcessGroupNCCL::CoalescedWorkNCCL> ProcessGroupNCCL::
    initCoalescedWork(
        const std::vector<c10::intrusive_ptr<Work>>& works,
        int rank,
        OpType opType) {
  std::vector<ProcessGroupNCCL::WorkNCCL> ncclWorks;
  ncclWorks.reserve(works.size());
  for (auto& work : works) {
    ncclWorks.push_back(*static_cast<ProcessGroupNCCL::WorkNCCL*>(work.get()));
  }
  return c10::make_intrusive<ProcessGroupNCCL::CoalescedWorkNCCL>(
      ncclWorks, rank, opType);
}

// Same as calling synchronize().
bool ProcessGroupNCCL::CoalescedWorkNCCL::wait(
    std::chrono::milliseconds timeout) {
  for (auto& w : works_) {
    w.wait(timeout);
  }
  // Always return true, because abort API is not implemented.
  return true;
}

static std::atomic<size_t> process_group_id = 0;

ProcessGroupNCCL::ProcessGroupNCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(options),
      ncclCommCounter_(0),
      traceKeyStart_(getTraceStartKey("NCCL", rank)),
      traceKeyEnd_(getTraceEndKey("NCCL", rank)),
      terminateProcessGroup_(false),
      terminateHeartbeatMonitorThread_(false),
      collectiveDebugInfoMode_(false),
      uid_(process_group_id++) {
  TORCH_CHECK_WITH(
      ValueError,
      at::cuda::getNumGPUs() != 0,
      "ProcessGroupNCCL is only supported with GPUs, no GPUs found!");
  blockingWait_ = parseEnvVarFlag(NCCL_BLOCKING_WAIT);
  asyncErrorHandling_ = static_cast<ErrorHandlingMode>(
      parseEnvVarIntDefault(NCCL_ASYNC_ERROR_HANDLING, 3 /*SkipCleanUp*/));
  desyncDebug_ = parseEnvVarFlag(NCCL_DESYNC_DEBUG) ||
      (dist_debug_level_ >= DebugLevel::Detail);
  heartbeat_ = 1ULL;
  monitorThreadEnabled_.store(parseEnvVarFlag(TORCH_NCCL_ENABLE_MONITORING));
  heartbeatTimeoutInSec_ = parseEnvVarIntDefault(
      TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC, 60 * 2 /*2 Mins*/);
#ifdef ENABLE_NCCL_ERROR_CHECKING
  enableTiming_.store(
      parseEnvVarFlag(NCCL_ENABLE_TIMING) || desyncDebug_ ||
      parseEnvVarIntDefault("TORCH_NCCL_TRACE_BUFFER_SIZE", 0) > 0);
#endif
  avoidRecordStreams_ = parseEnvVarFlag(TORCH_NCCL_AVOID_RECORD_STREAMS);
#ifdef NCCL_HAS_COMM_REGISTER
  useTensorRegisterAllocatorHook_ =
      parseEnvVarFlag(NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK);
  if (c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
          expandable_segments()) {
    useTensorRegisterAllocatorHook_ = false;
    LOG(INFO)
        << "[Rank " << rank_
        << "] disables NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK because it is not compatible with CUDA allocator expandable segments mode.";
  }
#endif

  if (blockingWait_) {
    if (asyncErrorHandling_ != NoHandling || desyncDebug_) {
      LOG(INFO) << "[Rank " << rank_ << "] NCCL_BLOCKING_WAIT and "
                << "NCCL_ASYNC_ERROR_HANDLING|NCCL_DESYNC_DEBUG"
                << "should not both be enabled. "
                << "Only NCCL_BLOCKING_WAIT is being used in this process.";
      asyncErrorHandling_ = NoHandling;
      desyncDebug_ = false;
    }
  } else {
    if (desyncDebug_ && asyncErrorHandling_ == NoHandling) {
      LOG(INFO) << "[Rank " << rank_
                << "] NCCL_DESYNC_DEBUG and NCCL_ASYNC_ERROR_HANDLING "
                << "must both be enabled. "
                << "Enabling NCCL_ASYNC_ERROR_HANDLING.";
      asyncErrorHandling_ = SkipCleanUp;
    }
  }

  if (parseEnvVarFlag(ENABLE_NCCL_HEALTH_CHECK)) {
    // Perform health check by initializing dummy communicators and destroying
    // them. This will help indicate any NCCL-related issues prior to the first
    // collective.
    // Run it in a separate thread and wait on CV to handle timeouts, since
    // majority of getNCCLComm failures are hangs.
    runHealthCheck();
  }

#ifdef ENABLE_NCCL_ERROR_CHECKING
  ncclCommWatchdogThread_ =
      std::thread(&ProcessGroupNCCL::ncclCommWatchdog, this);
#endif

  init();
  const std::string OFF = "OFF";
  const char* torch_distributed_debug =
      parseEnvVarString("TORCH_DISTRIBUTED_DEBUG", OFF.c_str());
  const char* nccl_debug = parseEnvVarString("NCCL_DEBUG", OFF.c_str());
  LOG(INFO) << "[Rank " << rank_
            << "] ProcessGroupNCCL initialization options: "
            << "NCCL version: " << getNcclVersion()
            << ", NCCL_ASYNC_ERROR_HANDLING: " << asyncErrorHandling_
            << ", NCCL_DESYNC_DEBUG: " << desyncDebug_
            << ", NCCL_ENABLE_TIMING: " << enableTiming_.load()
            << ", NCCL_BLOCKING_WAIT: " << blockingWait_
            << ", TIMEOUT(ms): " << options_->timeout.count()
            << ", USE_HIGH_PRIORITY_STREAM: "
            << options_->is_high_priority_stream
            << ", TORCH_DISTRIBUTED_DEBUG: "
            << std::string(torch_distributed_debug)
#ifdef NCCL_HAS_COMM_REGISTER
            << ", NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: "
            << useTensorRegisterAllocatorHook_
#endif
            << ", NCCL_DEBUG: " << std::string(nccl_debug)
            << ", ID=" << this->getID();

  RECORD_PARAM_COMMS(
      0, // seq
      this->getID(),
      rank, // rank
      "init", // colName
      0, // inSize
      0, // outSize
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      size_); // worldSize

  // Attach hooks to cache allocator to trigger the hooks whenever a traced
  // action is called. In the following hooks, we register a newly allocated
  // segment when SEGMENT_ALLOC action occurs, and deregister a segment when
  // SEGMENT_FREE action occurs.
  // We attach hooks only once at the first PG creation.
  if (useTensorRegisterAllocatorHook_ && !allocatorHooksAttached) {
    c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
        &cacheAllocatorRegisterHook);
    c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
        &cacheAllocatorDeregisterHook);
    allocatorHooksAttached = true;
  }

#ifdef USE_NCCL_WITH_UCC
  static c10::once_flag initialize_ucc_lib_flag;
  c10::call_once(initialize_ucc_lib_flag, [&] {
    uccLib_ = loadTorchUCC();
    if (uccLib_ != nullptr) {
      LOG(INFO) << "[Rank " << rank_ << "] torch_ucc.so loaded";
    }
  });

  if (uccLib_ != nullptr) {
    LOG(INFO) << "[Rank " << rank_ << "] torch_ucc.so loaded";
    typedef c10::intrusive_ptr<Backend> fn(
        const c10::intrusive_ptr<Store>& store, int rank, int size);
    auto createProcessGroupUCC =
        reinterpret_cast<fn*>(uccLib_->sym("createProcessGroupUCC"));
    if (createProcessGroupUCC != nullptr) {
      uccPG_ = createProcessGroupUCC(store, rank_, size_);
      LOG(INFO) << "[Rank " << rank_ << "] ProcessGroupUCC created.";
    }
  }
#endif
}

void ProcessGroupNCCL::runHealthCheck() {
  // Run health check in a separate thread and wait on CV to handle timeouts,
  // since majority of getNCCLComm failures are hangs.

  struct HealthCheckData {
    std::mutex healthCheckMutex;
    std::condition_variable healthCheckCv;
    bool healthCheckSuccess = false;
    std::exception_ptr healthCheckException;
  };

  HealthCheckData healthCheckData;
  auto t = std::thread([&healthCheckData, this]() {
    try {
      std::vector<at::Device> rankDevice = {getDeviceForRank(rank_)};
      const auto key = getKeyFromDevices(rankDevice);
      // OpType does not matter, only need to set to not go through send/recv
      // path.
      getNCCLComm(key, rankDevice, OpType::ALLREDUCE);
      // Now destroy the communicators and remove them from cache so we don't
      // use destroyed communicators.
      destroyNCCLComms(key);
      // Notify main thread the health check is complete.
      {
        std::lock_guard<std::mutex> lk(healthCheckData.healthCheckMutex);
        healthCheckData.healthCheckSuccess = true;
      }
      healthCheckData.healthCheckCv.notify_one();
    } catch (const std::exception& e) {
      // Populate exception ptr.
      healthCheckData.healthCheckException = std::current_exception();
      // Unblock waiting main thread which will report exception.
      healthCheckData.healthCheckCv.notify_one();
    } // Unknown exceptions will just cause the program to terminate.
  });
  // We don't need to join the thread, just need to verify health check via the
  // CV. Hence we detach the thread here.
  t.detach(); // NOLINT
  LOG(INFO) << "[Rank " << rank_ << "]"
            << " will wait up to " << options_->timeout.count()
            << " msec for NCCL health check to complete.";
  std::unique_lock<std::mutex> lock(healthCheckData.healthCheckMutex);
  healthCheckData.healthCheckCv.wait_for(
      lock, options_->timeout, [&healthCheckData]() {
        return healthCheckData.healthCheckSuccess;
      });

  if (healthCheckData.healthCheckException) {
    std::rethrow_exception(healthCheckData.healthCheckException);
  }
  // If there is no exception, the likely culprit is a timeout/hang which is how
  // most communicator init issues manifest themselves.
  TORCH_CHECK_WITH(
      DistBackendError,
      healthCheckData.healthCheckSuccess,
      "ProcessGroupNCCL: Health check failure: Failed to initialize NCCL communicator on rank ",
      rank_);
}

void ProcessGroupNCCL::setSequenceNumberForGroup() {
} // NCCL just starts sequence numbers at 0.

uint64_t ProcessGroupNCCL::getSequenceNumberForGroup() {
  return seq_;
}

void ProcessGroupNCCL::registerOnCompletionHook(
    std::function<void(std::shared_ptr<WorkInfo>)>&& hook) {
  TORCH_CHECK_WITH(
      DistBackendError,
      onCompletionHook_ == nullptr,
      "ProcessGroupNCCL OnCompletion hook already registered");

  TORCH_CHECK_WITH(
      ValueError,
      enableTiming_.load(),
      "ProcessGroupNCCL OnCompletion hook requires recording start and end "
      "events which require setting NCCL_ENABLE_TIMING environment variable. "
      "This is only available for NCCL version >= 2.4.");
  onCompletionHook_ = std::move(hook);
  onCompletionHookThread_ = std::thread(&ProcessGroupNCCL::runHookLoop, this);
}

// must release GIL when calling this method
void ProcessGroupNCCL::waitForPendingWorks() {
  // Reasoning about hook completion:
  // 1. waitForPendingWorks should be called after user code has finished
  // calling
  //    all collectives. This means, when we got here, all of the collectives
  //    are either in workMetaList_ or has been erased from workMetaList_.
  // 2. The watchdog thread grabs both locks to move Work object from the
  //    workMetaList_ to the completedWorkList_, and the hook thread only erases
  //    a Work object after the hook is returned. Therefore, after user code
  //    calls a collective, its Work object is either in workMetaList_ or in
  //    completedWorkList_ before it finishes.
  // 3. We have three threads and two locks.
  //      a. main thread (this function) grabs two locks atomically
  //      b. watchdog thread (workCleanupLoop function) always grabs
  //      workMetaListMutex_
  //         first and then grabs completedWorkListMutex_.
  //      c. hook thread (runHookLoop function) only grabs
  //      completedWorkListMutex_. Therefore, locks are always acquired in the
  //      same order and hence no deadlocks.
  while (true) {
    {
      std::lock(workMetaListMutex_, completedWorkListMutex_);
      std::lock_guard<std::mutex> lockWork(workMetaListMutex_, std::adopt_lock);
      std::lock_guard<std::mutex> lockHook(
          completedWorkListMutex_, std::adopt_lock);

      if (workMetaList_.empty() && completedWorkList_.empty()) {
        return;
      }
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(kWatchdogThreadSleepMillis));
  }
}

void ProcessGroupNCCL::enableCollectivesTiming() {
  enableTiming_.store(true);
}
void abortCommsFromMap(
    std::unordered_map<std::string, std::vector<std::shared_ptr<NCCLComm>>>&
        ncclCommsMap,
    const int rank,
    c10::optional<std::string> abortReason) {
  // The process may control multiple devices, loop through the communicators on
  // each device
  for (auto& it : ncclCommsMap) {
    auto& devName = it.first;
    auto& ncclComms = it.second;

    for (const auto& ncclComm : ncclComms) {
      ncclComm->ncclCommAbort(abortReason);
    }
    // Note that we don't remove the aborted communicators from the
    // cache. The reason is that if we do remove the communicator
    // from the cache, it is possible that a new collective operation
    // calls `ncclCommInitRank` to create a new communicator whereas
    // other ranks might have failed/timed out and didn't enter
    // `ncclCommInitRank`. As a result, when there is a failure on
    // a communicator the application receives an exception and its
    // their responsibility to destroy the process group and recreate
    // it to recover from errors.

    LOG(INFO) << "[Rank " << rank << "] Destroyed " << ncclComms.size()
              << "communicators on CUDA device " << devName;
  }
}

// Abort all communicators on this rank
void ProcessGroupNCCL::abort(c10::optional<std::string> abortReason) {
  // Remove record from global ncclCommDevIdxMapMutex before aboarting,
  // so that a new cache segment would not register to already aborded
  // communicators. Note that ncclCommDevIdxMap is a global container which may
  // contain other PG's communicators, thus we need to only erase communicators
  // for the current PG.
  ncclCommDevIdxMapMutex.lock();
  for (auto& it : devNCCLCommMap_) {
    auto& ncclComms = it.second;
    for (const auto& ncclComm : ncclComms) {
      ncclCommDevIdxMap.erase(ncclComm);
    }
  }
  ncclCommDevIdxMapMutex.unlock();

  std::lock_guard<std::mutex> lock(mutex_);
  abortCommsFromMap(devNCCLCommMap_, rank_, abortReason);
  abortCommsFromMap(inInitializationCommMap_, rank_, abortReason);
}

void ProcessGroupNCCL::shutdown() {
  // Don't join threads here since the purpose of this method is to abort all
  // communicators and signal the threads to exit. Joining on the threads could
  // potentially block and hence avoid it in this method.
  terminateProcessGroup_.store(true);

  std::string abortReason = c10::str("Process Group shutdown on rank ", rank_);
  abort(abortReason);

  workMetaListCV_.notify_one();
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
  terminateProcessGroup_.store(true);
  workMetaListCV_.notify_one();

#ifdef ENABLE_NCCL_ERROR_CHECKING
  if (ncclCommWatchdogThread_.joinable()) {
    ncclCommWatchdogThread_.join();
  }
#endif

  if (onCompletionHookThread_.joinable())
    onCompletionHookThread_.join();

  // Abort communicators after all threads have exited to avoid having the
  // threads dying due to aborted communicator and raising a SIGABRT
  std::string abortReason = c10::str("Process Group destroyed on rank ", rank_);
  abort(abortReason);

  // We need to wait for abort to finish before we can safely shut down
  // heartbeat monitoring thread.
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
#ifdef ENABLE_NCCL_ERROR_CHECKING
  if (ncclHeartbeatMonitorThread_.joinable()) {
    ncclHeartbeatMonitorThread_.join();
  }
#endif
}

void dump_debugging_info() {
  LOG(ERROR)
      << "No PGNCCL's watchdog heartbeat detected, so we are dumping debug info.";
  if (parseEnvVarIntDefault("TORCH_NCCL_TRACE_BUFFER_SIZE", 0) > 0) {
    // TODO: Find the right and proper way to dump the debug info.
    // We cannot print out debugging info directly.
    LOG(ERROR) << "nccl_trace: " << dump_nccl_trace();
  }
}

void ProcessGroupNCCL::terminateProcess(std::string errMsg) {
  // Logging with `FATAL`, after errMsg printed, it calls `std::abort()`
  // to terminate the program execution.
  LOG(FATAL) << errMsg;
}

void ProcessGroupNCCL::heartbeatMonitor() {
  uint64_t heartBeatCounter = 0ULL;
  while (true) {
    // This won't have any lock since this lock is only used here.
    // Please be aware that mutex `monitorMutex_` should not be used
    // somewhere else to avoid the deadlock.
    std::unique_lock<std::mutex> lock(monitorMutex_);
    if (monitorWakeUpCV_.wait_for(
            lock, std::chrono::seconds(heartbeatTimeoutInSec_), [&] {
              return terminateHeartbeatMonitorThread_.load();
            })) {
      // For the normal complete or user interception, monitorWakeUpCV_
      // will get notified, we early return and exit heartbeatMonitor.
      return;
    }

    // Check the heart beat of watchdog thread.
    auto heartbeat = heartbeat_;
    if (heartbeat != heartBeatCounter) {
      heartBeatCounter = heartbeat;
    } else {
      // No heartbeat increase detected and timeout.
      break;
    }
  }

  // In the timeout case and we now only dump the flight recorder
  // to std::out. Down the road, if we have more complicated or blocking
  // operations, we might need to use a side thread to do it.
  dump_debugging_info();

  // Create a error message reported from MonitorThread, so
  // we throw exception and make the whole process to be killed.
  const auto exitMsg = c10::str(
      "[Rank ",
      rank_,
      "] NCCL monitor thread timeout. Basically, this could ",
      "be due to CUDA or NCCL calls being unexpectedly blocking, ",
      "especially when your program enters a deadlock state in watchdog"
      "or destructors. If you see this error, please file a bug to pytorch.");

  // There are two possible cases for the watchdog thread exit:
  // Case one: desync report runs quickly, and it follows the step:
  // collective timeout -> desync -> exception handling -> destructors
  // -> set terminateHeartbeatMonitorThread_ -> notify monitorWakeUpCV_.
  // So the code either early returns above or will skip the sleep below.
  // Case two: desync might be slow or get stuck. Or we get stuck in
  // destructors, we will sleep for some time before calling std::abort() to
  // kill the whole process.
  if ((terminateProcessGroup_.load() || collectiveDebugInfoMode_.load()) &&
      !terminateHeartbeatMonitorThread_.load()) {
    // Leave another two mins for desync report generation or process group
    // destroy.
    std::this_thread::sleep_for(std::chrono::seconds(heartbeatTimeoutInSec_));
  }

  if (!terminateHeartbeatMonitorThread_.load()) {
    LOG(ERROR)
        << "monitoring thread detects no heartbeat and will kill the process!";
    terminateProcess(exitMsg);
  }
}

void ProcessGroupNCCL::ncclCommWatchdog() {
  try {
    VLOG(2) << "[Rank " << rank_ << "] NCCL watchdog thread started!";
    if (monitorThreadEnabled_.load()) {
      ncclHeartbeatMonitorThread_ =
          std::thread(&ProcessGroupNCCL::heartbeatMonitor, this);
    }
    workCleanupLoop();
    VLOG(2) << "[Rank " << rank_
            << "] NCCL watchdog thread terminated normally";
  } catch (std::exception& e) {
    if (std::string(e.what()).find("driver shutting down") !=
        std::string::npos) {
      LOG(INFO)
          << "[Rank " << rank_
          << "] main process destroyed cuda before watchdog loop exited, terminating watchdog."
          << " (Watchdog caught exception: " << e.what();

    } else {
      // Append error message reported from workCleanupLoop
      const auto exitMsg = c10::str(
          "[Rank ",
          rank_,
          "] NCCL watchdog thread terminated with exception: ",
          e.what());
      LOG(ERROR) << exitMsg;
      // TODO(whc) clean up the rethrow - why is it stored in a class var and
      // rethrown?
      watchDogException_ =
          std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
      std::rethrow_exception(watchDogException_);
    }
  } catch (...) {
    const auto exitMsg = c10::str(
        "[Rank ",
        rank_,
        "] NCCL watchdog thread terminated with exception: unknown");
    LOG(ERROR) << exitMsg;
    watchDogException_ =
        std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
    std::rethrow_exception(watchDogException_);
  }
}

void ProcessGroupNCCL::logWorkStart(WorkNCCL& work) {
  if (work.startTraceUpdated_)
    return;

  if (terminateProcessGroup_.load() || storeError_)
    return;

  work.startTraceUpdated_ = true;
  storeError_ = !c10d::traceUpdate(
      store_, traceKeyStart_, work.seq_, opTypeToString(work.opType_));
}

void ProcessGroupNCCL::logWorkEnd(WorkNCCL& work) {
  if (terminateProcessGroup_.load() || storeError_)
    return;

  // In case the start of the work hasn't been logged
  if (!work.startTraceUpdated_) {
    logWorkStart(work);
  }

  storeError_ = !c10d::traceUpdate(
      store_, traceKeyEnd_, work.seq_, opTypeToString(work.opType_));
}

std::string ProcessGroupNCCL::getNCCLWatchdogDebugInfo() {
  return retrieveDesyncReport(store_, "NCCL", rank_, size_);
}

void ProcessGroupNCCL::workCleanupLoop() {
  bool done = false;

  std::list<ProcessGroupNCCL::WorkNCCL> completedWorkList;
  while (!done || !terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(workMetaListMutex_);
    // We busy-poll the work vector every kWatchdogThreadSleepMillis
    // milliseconds as long as the atomic is True.
    workMetaListCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool { return terminateProcessGroup_.load(); });
    // Bump up heart beat by one.
    heartbeat_++;

    for (auto it = workMetaList_.begin(); it != workMetaList_.end();
         /* no increment */) {
      auto& work = *it;
      work.checkAndSetException();
      bool timedOut = work.checkTimeout();

      // If work hits an exception (either an error or timeout)
      if (work.exception()) {
        if (SHOULD_CLEAN_UP(asyncErrorHandling_)) {
          // Abort work and corresponding communicators
          work.abort();
          // PG level abort, which would abort all other communicators on this
          // rank
          abort();
        }
        // Report desync state in case of timeout
        if (desyncDebug_ && timedOut) {
          try {
            // Set shutdown mode, so the heartbeat monitor thread will not abort
            // process immediately.
            collectiveDebugInfoMode_.store(true);
            auto desyncMsg = getNCCLWatchdogDebugInfo();
            LOG(ERROR) << desyncMsg;
          } catch (const std::exception& e) {
            LOG(ERROR) << "Failed to retrieve NCCL_DESYNC_DEBUG report. "
                       << " Please file an issue. Error: " << e.what();
          } catch (...) {
            LOG(ERROR)
                << "Failed to rerieve NCCL_DESYNC_DEBUG report with unknown error."
                << " Please file an issue.";
          }
        }
        // Throw exception
        work.handleException(asyncErrorHandling_);
      }

      // Work status logging for desync debug
      if (desyncDebug_) {
        if (work.isStarted()) {
          logWorkStart(work);
        }
        if (work.isCompleted()) {
          logWorkEnd(work);
        }
      }

      // Clean up completed work
      if (work.isCompleted()) {
        NCCLTraceBuffer::get()->complete(work.trace_id_);
        if (onCompletionHook_) {
          // Move Work object to completedWorkList_ to be consumed by the hook
          // thread
          {
            const std::lock_guard<std::mutex> lock(completedWorkListMutex_);
            completedWorkList_.splice(
                completedWorkList_.end(), workMetaList_, it++);
          }
          completedWorkListCV_.notify_one();
        } else {
          it = workMetaList_.erase(it);
        }
        at::cuda::CUDAGraph::dec_pending_event_queries();
      } else {
        // Increment the iterator if the current WorkNCCL object is not
        // completed.
        ++it;
      }
    }

    done = workMetaList_.empty();
  }
}

void ProcessGroupNCCL::runHookLoop() {
  bool done = false;
  while (!done || !terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(completedWorkListMutex_);
    // We busy-poll the work vector every kWatchdogThreadSleepMillis
    // milliseconds as long as the atomic is True.
    completedWorkListCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool {
          return !completedWorkList_.empty() || terminateProcessGroup_.load();
        });

    try {
      for (auto it = completedWorkList_.begin(); it != completedWorkList_.end();
           /* no increment */) {
        const WorkNCCL& work = *it;
        // Hook might grab GIL, unlock first to prevent deadlock
        lock.unlock();

        auto timeStarted =
            std::chrono::system_clock::now() +
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                work.workStartTime_ - std::chrono::steady_clock::now());
        onCompletionHook_(std::make_shared<WorkInfo>(
            work.retrieveOpType(), // OpType
            timeStarted, // timeStarted
            std::chrono::system_clock::now(), // timeFinished
            std::chrono::duration<float, std::milli>(
                work.getDuration()) // activeDuration
            ));

        lock.lock();
        it = completedWorkList_.erase(it);
      }
    } catch (std::exception& e) {
      if (std::string(e.what()).find("driver shutting down") !=
          std::string::npos) {
        LOG(INFO)
            << "[Rank " << rank_
            << "] main process destroyed cuda before runHookLoop exited, terminating runHookLoop."
            << " (runHookLoop caught exception: " << e.what();

      } else {
        // PythonOnCompletionHook has already extracted Python exception message
        // and wrapped it with a cpp one. So we no longer need to acquire GIL
        // here.
        const auto errorStr = c10::str(
            "Caught exception on rank ",
            rank_,
            " while running onCompletion hook for ProcessGroupNCCL: ",
            e.what(),
            ". Aborting all communicators.");

        // No need to call abort() on WorkNCCL here as that collective has
        // already finished successfully at this point. We just need to abort
        // the process Abort all NCCL Communicators on this ProcessGroupNCCL
        // instance.
        abort(errorStr);
      }
    }

    // Lock is still acquired at this point
    done = completedWorkList_.empty();
  }
}

std::exception_ptr ProcessGroupNCCL::WorkNCCL::checkForNCCLErrors(
    const std::vector<std::shared_ptr<NCCLComm>>& ncclComms) const {
  return checkForNCCLErrorsInternal(ncclComms);
}

std::exception_ptr ProcessGroupNCCL::checkForNCCLErrors(
    const std::vector<std::shared_ptr<NCCLComm>>& ncclComms) {
  return checkForNCCLErrorsInternal(ncclComms);
}

std::exception_ptr ProcessGroupNCCL::checkForNCCLErrorsInternal(
    const std::vector<std::shared_ptr<NCCLComm>>& ncclComms) {
  for (const auto& ncclComm : ncclComms) {
    // Prioritize commFailureReason over checkForNcclError() result if
    // commFailureReason is set.
    auto commFailureReason = ncclComm->getNcclCommFailureReason();
    if (commFailureReason != c10::nullopt) {
      return std::make_exception_ptr(C10_BUILD_ERROR(
          DistBackendError,
          c10::str(
              "NCCL communicator encountered error set by ProcessGroupNCCL: ",
              *commFailureReason)));
    }
    ncclResult_t ncclAsyncErr = ncclComm->checkForNcclError();
    if (ncclAsyncErr != ncclSuccess) {
      return std::make_exception_ptr(C10_BUILD_ERROR(
          DistBackendError,
          "NCCL error: " + ncclGetErrorWithVersion(ncclAsyncErr) + "\n" +
              getNcclErrorDetailStr(ncclAsyncErr)));
    }
  }

  return nullptr;
}

void ProcessGroupNCCL::broadcastUniqueNCCLID(
    ncclUniqueId* ncclID,
    bool isSingleP2POp,
    const std::string& p2pKey,
    int p2pRank) {
  // For collective operations:
  // For every NCCL communicator that we create we need to broadcast
  // a unique ID from rank 0 to all other ranks. This broadcast is
  // done by rank 0 setting a key in the store and all other ranks
  // retrieving the contents of that key. A single process group
  // may create multiple NCCL communicators, so we use a sequence
  // number to differentiate between them.
  // For single point-to-point operations:
  // The sequence number will only be increased on 2 out of all the
  // processes in a Process Group. So all following collective
  // operations will see different sequence numbers which will cause
  // runtime errors. To avoid that, use the src:target pair instead
  // of sequence number for p2p communications.

  std::string storeKey;
  if (!isSingleP2POp) {
    storeKey = std::to_string(ncclCommCounter_++);
  } else {
    storeKey = p2pKey;
  }
  if (rank_ == 0 || (isSingleP2POp && p2pRank == 0)) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(ncclID),
        reinterpret_cast<uint8_t*>(ncclID) + NCCL_UNIQUE_ID_BYTES);
    store_->set(storeKey, vec);
  } else {
    try {
      auto vec = store_->get(storeKey);
      TORCH_CHECK_WITH(
          DistBackendError,
          vec.size() == NCCL_UNIQUE_ID_BYTES,
          "Invalid size for ncclUniqueId");
      std::memcpy(ncclID, vec.data(), vec.size());
    } catch (const std::exception& e) {
      std::string exceptionMsg = c10::str(
          "[",
          rank_,
          "] is setting up NCCL communicator and "
          "retrieving ncclUniqueId from [0] via c10d key-value store by key '",
          storeKey,
          "', but store->get('",
          storeKey,
          "') got error: ");
      C10_THROW_ERROR(
          DistBackendError,
          exceptionMsg + e.what() +
              ". This may indicate a possible application crash on rank 0 or a network set up issue.");
    } catch (...) {
      C10_THROW_ERROR(
          DistBackendError,
          c10::str(
              "Unknown exception while [",
              rank_,
              "] is setting up NCCL communicator and "
              "retrieving ncclUniqueId from [0] via c10d key-value store by key '",
              storeKey,
              "'",
              ". This may indicate a possible application crash on rank 0 or a network set up issue."));
    }
  }
}

void ProcessGroupNCCL::destroyNCCLComms(const std::string& devNCCLCommMapKey) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (devNCCLCommMap_.find(devNCCLCommMapKey) == devNCCLCommMap_.end()) {
    TORCH_INTERNAL_ASSERT(
        false,
        "Expected to find key ",
        devNCCLCommMapKey,
        " in NCCL communicator map.");
  }
  std::vector<std::shared_ptr<NCCLComm>>& ncclComms =
      devNCCLCommMap_[devNCCLCommMapKey];
  // Loop through communicators and call ncclCommAbort.
  for (const auto& comm : ncclComms) {
    // ncclCommDestroy(comm->getNcclComm()) results in segfault when PG is being
    // destroyed, so using ncclCommAbort here.
    comm->ncclCommAbort();
  }
  // Remove communicators from the cache.
  devNCCLCommMap_.erase(devNCCLCommMapKey);
  // Clear used device indices.
  usedDeviceIdxs_.clear();

  ncclCommDevIdxMapMutex.lock();
  for (const auto& comm : ncclComms) {
    ncclCommDevIdxMap.erase(comm);
  }
  ncclCommDevIdxMapMutex.unlock();
}

std::vector<std::shared_ptr<NCCLComm>>& ProcessGroupNCCL::getNCCLComm(
    const std::string& devicesKey,
    const std::vector<at::Device>& devices,
    OpType opType,
    int p2pRank,
    bool isSendRecvSelf) {
  // Sanity check
  if (devicesKey.empty()) {
    C10_THROW_ERROR(
        DistBackendError,
        "Not able to create/get the NCCL Communicator since "
        "the GPU devices are not known");
  }

  for (auto& device : devices) {
    usedDeviceIdxs_.insert(device.index());
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (devNCCLCommMap_.find(devicesKey) != devNCCLCommMap_.end()) {
      // Reuse the cached communicator if there is one.
      return devNCCLCommMap_[devicesKey];
    }
  }

  // NCCL communicator not cached, create a new entry
  std::vector<std::shared_ptr<NCCLComm>> ncclComms;
  ncclComms.resize(devices.size());

  // Create the unique NCCL ID and broadcast it
  ncclUniqueId ncclID;

  // For batch_isend_irecv, ncclGroupStart() would be called upfront
  bool batchP2P = ncclActiveGroupCounter_ > 0;
  bool singleP2POp = isP2POp(opType, batchP2P);
  // For point-to-point communication, lower rank of the two will get unique id.
  if (rank_ == 0 || (singleP2POp && p2pRank == 0)) {
    C10D_NCCL_CHECK(ncclGetUniqueId(&ncclID), c10::nullopt);
  }

  // For point-to-point communication on the same process, don't need broadcast.
  if (!isSendRecvSelf) {
    // Broadcast so that each process can have a unique NCCL ID
    broadcastUniqueNCCLID(&ncclID, singleP2POp, devicesKey, p2pRank);
  }

  at::cuda::OptionalCUDAGuard gpuGuard;

  std::vector<at::cuda::CUDAStream> streamVal;
  streamVal.reserve(devices.size());

  // [Group Start/End Note] This is used to ensure that nccl communicator will
  // be created before communication primitives are called. Let's look at this
  // example: Using the batch_isend_irecv to send a tensor to a target process.
  // On the sender side, the corresponding underlying NCCL calls will look like
  //   ncclGroupStart() // This is in batch_isend_irecv
  //   ncclGroupStart() // This is [Note 1]
  //   ncclCommInitRank() // Inside NCCLComm::create
  //   ncclSend()
  //   ncclGroupEnd() // This is [Note 2]
  //   ncclGroupEnd() // This is in batch_isend_irecv
  // With this pattern, the nccl communicator will be created in the last
  // ncclGroupEnd which means when ncclSend is processed, the passed
  // communicator argument is NULL which will lead to runtime error. So we need
  // to "close" all active nccl groups to ensure nccl communicator is actually
  // created before encountering any communication calls. This is why we need
  // the following for loop.
  for (const auto i : c10::irange(ncclActiveGroupCounter_)) {
    (void)i;
    // comms have not been initiated yet, so can only check in blocking-way
    C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
  }

  // [Note 1] Create the NCCL communicators for each GPU
  C10D_NCCL_CHECK(ncclGroupStart(), c10::nullopt);

  for (const auto i : c10::irange(devices.size())) {
    // GPU world size and GPU rank
    int numRanks, rank;

    if (!singleP2POp) {
      // Collective, all-to-all, or batch P2P
      numRanks = getSize() * devices.size();
      rank = getRank() * devices.size() + i;
    } else if (isSendRecvSelf) {
      // Same process send and recv.
      numRanks = 1;
      rank = 0;
    } else {
      // For single point-to-point operation, there are only 2 processes
      // involved so the GPU rank is either 0 or 1.
      numRanks = 2;
      rank = p2pRank;
    }
    // Get the device index
    int deviceIndex = devices[i].index();

    gpuGuard.set_index(deviceIndex);
#ifdef NCCL_HAS_COMM_NONBLOCKING
    ncclComms[i] = NCCLComm::create(numRanks, rank, ncclID, options_->config);
#else
    ncclComms[i] = NCCLComm::create(numRanks, rank, ncclID);
#endif

    // Creates the NCCL streams
    streamVal.push_back(
        at::cuda::getStreamFromPool(options_->is_high_priority_stream));
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    inInitializationCommMap_.emplace(devicesKey, ncclComms);
  }

  // [Note 2 ]
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
#else
  if (!nccl_use_nonblocking()) {
    C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
  } else {
    C10D_NCCL_CHECK_TIMEOUT_GROUPEND(ncclGroupEnd(), ncclComms, c10::nullopt);
  }
#endif

  // At this point NCCL should have been initialized, hence we can accurately
  // get the env value even if NCCL sets it by reading from nccl.conf file
  if (getRank() == 0) {
    LOG(INFO) << "NCCL_DEBUG: " << parse_env("NCCL_DEBUG");
  }

  // See [Group Start/End Note]
  for (const auto i : c10::irange(ncclActiveGroupCounter_)) {
    (void)i;
    C10D_NCCL_CHECK(ncclGroupStart(), c10::nullopt);
  }

  ncclStreams_.emplace(devicesKey, std::move(streamVal));

  // Note: these events are created with the (default) cudaEventDisableTiming
  // flag This flag provides the best performance when used with
  // cudaStreamWaitEvent() and cudaEventQuery(). Since we here don't measure the
  // performance using cudaEvent, this should be set.
  ncclEvents_.emplace(
      std::piecewise_construct,
      std::make_tuple(devicesKey),
      std::make_tuple(devices.size()));

  // Hold the lock before modifying the cache.
  std::lock_guard<std::mutex> lock(mutex_);

  // Record the communicators based on ncclUniqueId.
  ncclIdToCommMap_.emplace(buildNcclUniqueIdStr(ncclID), ncclComms);

  // Move the NCCL resource to cache
  auto it = inInitializationCommMap_.find(devicesKey);
  // A previous thread could've already removed devicesKey from
  // inInitializationCommMap_ and added it to devNCCLCommMap_
  if (it != inInitializationCommMap_.end()) {
    devNCCLCommMap_.emplace(devicesKey, std::move(it->second));
    inInitializationCommMap_.erase(devicesKey);

    // Now ncclComms are fully initialized.
    // Register all active CUDA memory segments in cache allocator to
    // the new NCCL communicators
    if (useTensorRegisterAllocatorHook_) {
      auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
      // Register the segment to a new NCCL communicator if on the same device
      for (const auto& segmentInfo : snapshot.segments) {
        for (const auto i : c10::irange(devices.size())) {
          if (segmentInfo.device != devices[i].index())
            continue;
          ncclComms[i]->registerSegment(
              reinterpret_cast<void*>(segmentInfo.address),
              segmentInfo.total_size);
        }
      }

      // Record the mapping between ncclComm and device index so that later
      // register hook can register a newly allocated segment to communicators
      // on the same device.
      // NOTE: we need remove the communicator from this map when it is
      // destroyed, otherwise may register onto an invalid communicator.
      ncclCommDevIdxMapMutex.lock();
      for (const auto i : c10::irange(devices.size())) {
        ncclCommDevIdxMap.emplace(ncclComms[i], devices[i].index());
      }
      ncclCommDevIdxMapMutex.unlock();
    }
  }

  it = devNCCLCommMap_.find(devicesKey);
  TORCH_INTERNAL_ASSERT(
      it != devNCCLCommMap_.end(), "Communicators not populated in cache!");
  return it->second;
}

namespace {

// Check validity of tensor
void check_gpu_single_tensor(const at::Tensor& tensor) {
  if (!tensor.is_cuda() || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be CUDA and dense");
  }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
  }
}

// Checks that all `tensors' have the same type and shape and reside on distinct
// GPUs.
// TODO: test_c10d_nccl.py should consider adding tests for the error conditions
// here, ie, that deliberately pass invalid tensors and check the right
// exception is thrown.
void check_gpu_tensors_different_devices(
    const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    C10_THROW_ERROR(ValueError, "Tensor list must be nonempty");
  }
  if (tensors.size() > static_cast<size_t>(at::cuda::getNumGPUs())) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor list mustn't be larger than the number of available GPUs");
  }

  const auto& first = tensors.front();

  // Set for ensuring that tensors are on separate devices.
  std::unordered_set<decltype(first.get_device())> usedDevices;
  usedDevices.reserve(tensors.size());

  for (const auto& t : tensors) {
    if (!t.is_cuda() || t.is_sparse()) {
      C10_THROW_ERROR(ValueError, "Tensors must be CUDA and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      C10_THROW_ERROR(TypeError, "Tensors must have identical type");
    }
    if (t.sizes() != first.sizes()) {
      C10_THROW_ERROR(ValueError, "Tensors must have identical size");
    }
    if (t.strides() != first.strides()) {
      C10_THROW_ERROR(ValueError, "Tensors must have identical strides");
    }
    if (!t.is_contiguous(t.suggest_memory_format())) {
      C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
    }
    const auto inserted = usedDevices.insert(t.get_device()).second;
    if (!inserted) {
      C10_THROW_ERROR(ValueError, "Tensors must be on distinct GPU devices");
    }
  }
}

// Checks that all `tensors' have the same type and shape and reside on the same
// GPU.
// TODO: test_c10d_nccl.py should consider adding tests for the error conditions
// here, ie, that deliberately pass invalid tensors and check the right
// exception is thrown. The "Expected list of tensors on the same device"
// condition may be a challenge because the test would need to pass tensors on
// different devices in the same process.
int64_t check_gpu_tensors_same_device(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    C10_THROW_ERROR(ValueError, "Tensor list must be nonempty");
  }

  const auto& first = tensors.front();

  int64_t total_numel = 0;
  for (const auto& t : tensors) {
    if (!t.is_cuda() || t.is_sparse()) {
      C10_THROW_ERROR(ValueError, "Tensors must be CUDA and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      C10_THROW_ERROR(TypeError, "Tensors must have identical type");
    }
    if (!t.is_non_overlapping_and_dense()) {
      C10_THROW_ERROR(ValueError, "Tensors must be non-overlapping and dense");
    }
    // If we're in this function, the user called a _coalesced collective
    // on a set of tensors with potentially different sizes and strides.
    // Therefore, we don't check for matching sizes and strides,
    // but we do double-check tensors are on the same device.
    TORCH_CHECK_WITH(
        ValueError,
        t.get_device() == tensors[0].get_device(),
        "Expected list of tensors on the same device");
    total_numel += t.numel();
  }

  return total_numel;
}

bool check_same_size(const std::vector<at::Tensor>& input_tensors) {
  for (const auto& input_tensor : input_tensors) {
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  return true;
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size) {
  if (tensor_lists.size() != other.size()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor list operands to scatter/gather must have the same length");
  }
  const auto num_devices = tensor_lists.size();

  std::vector<at::Tensor> flattened;
  flattened.resize(num_devices);

  for (const auto i : c10::irange(size_t{}, num_devices)) {
    if (tensor_lists[i].size() != world_size * num_devices) {
      C10_THROW_ERROR(
          ValueError,
          c10::str(
              "Tensor list input to scatter/gather must match number of collective participants ",
              "but got ",
              tensor_lists[i].size(),
              " inputs",
              " with world_size ",
              world_size,
              " and ",
              num_devices,
              " devices."));
    }

    // Only check device match for the first tensor in the list; the call to
    // newLikeFlat() below will check the rest.
    if (tensor_lists[i].front().get_device() != other[i].get_device()) {
      C10_THROW_ERROR(
          ValueError,
          "Corresponding input/output tensors to scatter/gather must all reside"
          " on the same device");
    }

    for (const auto& t : tensor_lists[i]) {
      if (t.numel() != other[i].numel()) {
        C10_THROW_ERROR(
            ValueError,
            "All tensor operands to scatter/gather must have the same number of elements");
      }
    }
    // Flatten the tensors (from all ranks) into a single big tensor.
    flattened[i] = newLikeFlat(tensor_lists, i);
  }
  return flattened;
}

} // namespace

c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> ProcessGroupNCCL::initWork(
    std::vector<at::Device> devices,
    int rank,
    OpType opType,
    const char* profilingTitle,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs) {
  auto r = c10::make_intrusive<ProcessGroupNCCL::WorkNCCL>(
      devices,
      rank,
      opType,
      seq_,
      profilingTitle,
      profilingTitle != nullptr ? c10::optional<std::vector<at::Tensor>>(inputs)
                                : c10::nullopt,
      desyncDebug_,
      enableTiming_.load());
  r->trace_id_ = NCCLTraceBuffer::get()->record(
      uid_,
      seq_,
      profilingTitle,
      inputs,
      outputs,
      r->ncclStartEvents_.get(),
      r->ncclEndEvents_.get());
  return r;
}

std::vector<at::Tensor> ProcessGroupNCCL::WorkNCCL::result() {
  return *outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupNCCL::WorkNCCL::
    getFuture() {
  return future_;
}

float ProcessGroupNCCL::WorkNCCL::getDuration() const {
  TORCH_CHECK(timingEnabled_, "getDuration only works if timing was enabled")
  TORCH_CHECK(
      ncclStartEvents_->size() == 1,
      "getDuration only works for single device per ProcessGroup.");
  TORCH_CHECK(
      ncclEndEvents_->size() == 1,
      "getDuration only works for single device per ProcessGroup.");
  TORCH_CHECK(
      (*ncclEndEvents_)[0].query(),
      "getDuration can only be called after work is succeeded.")
  return (*ncclStartEvents_)[0].elapsed_time((*ncclEndEvents_)[0]);
}
uint64_t ProcessGroupNCCL::WorkNCCL::getSequencenumber() const {
  return seq_;
}

void ProcessGroupNCCL::workEnqueue(
    c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL> work) {
  if (!terminateProcessGroup_.load()) {
    std::lock_guard<std::mutex> lock(workMetaListMutex_);
    // Avoid view tensors to be processed in cleanup thread.
    // View tensors' destruction invokes autograd_meta, which
    // needs to be destructed in user thread. Otherwise will
    // get deadlock. Here we enqueue work without outputs_.
    workMetaList_.emplace_back(*work);
  }
}

ProcessGroupNCCL::Options::Options(bool is_high_priority_stream)
    : Backend::Options(NCCL_BACKEND_NAME, kProcessGroupNCCLDefaultTimeout),
      is_high_priority_stream(is_high_priority_stream) {}

static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;

void ProcessGroupNCCL::startCoalescing() {
  coalescedDevices_.clear();
  coalescedComms_.clear();
  coalescing_state_ |= CoalActive;
  groupStart();
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::endCoalescing() {
  if (!nccl_use_nonblocking() ||
      coalescedComms_.size() == 0) { // There is no actual work being coalesced
    groupEnd();
  } else {
    // `coalescedComms_` should have same set of comms across collectives
    auto comms = coalescedComms_[0];
    groupEndNonblocking(comms);
  }

  coalescing_state_ = 0;

  if (coalescedDevices_.size() == 0) {
    // There is no actual work being coalesced
    return nullptr;
  }

  // `coalescedDevices_` should have same set of devices across collectives
  auto devices = coalescedDevices_[0];

  // Create Work object
  auto work = initWork(devices, rank_, OpType::COALESCED, "nccl:coalesced");

  // Record stream event
  // `getKeyFromDevices` is how we get keys for both collectives and batch P2P
  const auto key = getKeyFromDevices(devices);
  auto& ncclStreams = ncclStreams_[key];
  // TODO(eqy): is this still necessary if avoidRecordStreams_ is set?
  for (const auto i : c10::irange(devices.size())) {
    auto& devEvent = (*work->ncclEndEvents_)[i];
    devEvent.record(ncclStreams[i]);
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams_;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;
  if (avoidRecordStreams_) {
    // other functions expect an initialized ptr if avoidRecordStreams_ is set
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>();
  }
  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();

  if ((coalescing_state_ & CoalColl) &&
      capture_status == c10::cuda::CaptureStatus::None) {
    workEnqueue(work);
    // TODO: it seems we never enqueue work for single send/recv or batch P2P,
    // see the `pointToPoint` function. This should be fixed. Otherwise, we risk
    // not being able to abort hanged P2P ops.
  }

  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType,
    const char* profilingTitle,
    bool avoidRecordStreams) {
  // Environment setting by the user may add onto collective call's option
  avoidRecordStreams |= avoidRecordStreams_;
  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();
  errorIfCapturingNonCapturableNCCL(capture_status);

  // Bump collective counter
  seq_++;

  // Currently, the API permits two scenarios where inputs.size() and
  // outputs.size() are > 0.
  // 1. If the call was a _coalesced call, all inputs must be on the same
  // device.
  //    The group of nccl calls applies the collective separately to each input,
  //    but the group as a whole should be efficient, and might even execute as
  //    a single fused kernel.
  // 2. If the call was a _multigpu call, all inputs must be on different
  // devices.
  //    The nccl group applies the collective across them (eg, if the collective
  //    is an allreduce, the output on each device contains contributions summed
  //    across `inputs' tensors).
  const auto devices = getDeviceList(inputs);
  const bool inputs_same_dev = (devices.size() == 1);
  const auto key = getKeyFromDevices(devices);
  auto& ncclComms = getNCCLComm(key, devices, opType);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalColl;
    coalescedDevices_.push_back(devices);
    coalescedComms_.push_back(ncclComms);
  }

  // Used many times below, so we stash the unordered_map lookup
  auto& ncclStreams = ncclStreams_[key];

  // First let NCCL streams wait for input tensors allocation streams
  syncStreams(devices, ncclEvents_[key], ncclStreams);

  // Work itself will create the CUDA events on all GPUs of tensors
  bool can_profile = outputs.size() == 1;

  auto work = initWork(
      devices,
      rank_,
      opType,
      can_profile ? profilingTitle : nullptr,
      inputs,
      outputs);

  // Store references to outputs to be used by WorkNCCL::result and operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  if (avoidRecordStreams) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>(inputs);
  }

  at::cuda::OptionalCUDAGuard gpuGuard;

  // Start event should only be recorded before the ncclGroupStart()
  if (work->timingEnabled_) {
    for (const auto i : c10::irange(devices.size())) {
      at::cuda::CUDAStream& ncclStream = ncclStreams[i];
      (*work->ncclStartEvents_)[i].record(ncclStream);
    }
  }

  pre(ncclStreams, work);

  std::vector<void*> comms_;
  if (nccl_use_nonblocking()) {
    for (const auto i : c10::irange(inputs.size())) {
      decltype(i) stream_comm_i = (inputs_same_dev ? 0 : i);
      comms_.push_back((void*)ncclComms[stream_comm_i]->getNcclComm());
    }
  }

  {
    torch::cuda::nccl::AutoNcclGroup nccl_group_guard(
        comms_, nccl_use_nonblocking());
    for (const auto i : c10::irange(inputs.size())) {
      if (!inputs_same_dev || (inputs_same_dev && i == 0)) {
        gpuGuard.set_index(devices[i].index());
      }
      decltype(i) stream_comm_i = (inputs_same_dev ? 0 : i);
      auto& ncclStream = ncclStreams[stream_comm_i];
      auto& ncclComm = ncclComms[stream_comm_i];
      // Both `inputs' and `outputs' are created on a worker stream and used in
      // different ncclStreams.  Hence, both must record the ncclStream to
      // prevent being freed before the collective finishes.
      //
      // We only record `inputs' here, and leave recording `outputs' to `fn' for
      // operations where `inputs' and `outputs' are not the same.
      //
      // See [Sync Streams].
      if (!avoidRecordStreams) {
        if (!inputs[i].is_sparse()) {
          c10::cuda::CUDACachingAllocator::recordStream(
              inputs[i].storage().data_ptr(), ncclStream);
        } else {
          // for sparse input case record streams on both index and value
          // tensors
          c10::cuda::CUDACachingAllocator::recordStream(
              inputs[i].values().storage().data_ptr(), ncclStream);
          c10::cuda::CUDACachingAllocator::recordStream(
              inputs[i].indices().storage().data_ptr(), ncclStream);
        }
      }
#ifndef NCCL_HAS_COMM_NONBLOCKING
      C10D_NCCL_CHECK(
          fn(inputs[i], outputs[i], ncclComm->getNcclComm(), ncclStream),
          ncclComm->getNcclCommFailureReason());
#else
      C10D_NCCL_CHECK_TIMEOUT(
          fn(inputs[i], outputs[i], ncclComm->getNcclComm(), ncclStream),
          ncclComm->getNcclComm(),
          ncclComm->getNcclCommFailureReason());
#endif
    }
  }
  post(ncclStreams, work);

  // End event should only be recorded after the ncclGroupEnd()
  for (const auto i : c10::irange(devices.size())) {
    at::cuda::CUDAStream& ncclStream = ncclStreams[i];
    if (!coalescing_state_) {
      (*work->ncclEndEvents_)[i].record(ncclStream);
    }
    work->ncclComms_[i] = ncclComms[i];
  }

  {
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStreams);
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in CUDA
    // future blocks the stream this callback runs on the corresponding
    // ncclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;
  // Record size info for debug. We only record the size on the first device as
  // multi-device per process is deprecated
  work->numelIn_ = inputs[0].numel();
  work->numelOut_ = outputs[0].numel();

  // Notify graphs before we check the capture status preemptively
  at::cuda::CUDAGraph::inc_pending_event_queries();

  if (!coalescing_state_ && capture_status == c10::cuda::CaptureStatus::None) {
    workEnqueue(work);
  } else {
    at::cuda::CUDAGraph::dec_pending_event_queries();
  }

  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupNCCL::pointToPoint(
    std::vector<at::Tensor>& tensors,
    Fn fn,
    int peer,
    OpType opType,
    PreProcess pre,
    PostProcess post,
    const char* profilingTitle) {
  // avoidRecordStreams_ note:
  // send, recv, and irecv should be ok with avoidRecordStreams,
  // However, for isend, I don't think the API requires the user
  // to wait() on the returned handle, so ProcessGroupNCCL can't know
  // when it's safe to release the input back to the allocator,
  // and the present call has no way to know it's not an isend.
  // Therefore, we warn and fall back to the typical recordStream logic:
  TORCH_WARN_ONCE(
      avoidRecordStreams_,
      "TORCH_NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point "
      "collectives.");

  // Bump sequence number, updated in collective() as well
  seq_++;

  const auto devices = getDeviceList(tensors);
  std::string key;
  int p2pRank = 0, p2pTargetRank = 0;
  bool isSendRecvSelf = false;
  // For batch_isend_irecv, ncclGroupStart() would be called upfront
  bool batchP2P = ncclActiveGroupCounter_ > 0;
  if (batchP2P) {
    // For batch P2P, we need to treat it like a collective when selecting
    // communicator, because other ranks can call into this batch other than my
    // rank and my peer
    key = getKeyFromDevices(devices);
    p2pRank = rank_;
    p2pTargetRank = peer;
  } else {
    // For single P2P, preserve the old two-rank behavior (to avoid perf diff)
    key = getKeySendRecv(rank_, peer);
    p2pRank = rank_ <= peer ? 0 : 1;
    isSendRecvSelf = rank_ == peer;
    p2pTargetRank = isSendRecvSelf ? 0 : 1 - p2pRank;
  }
  auto& ncclComms = getNCCLComm(key, devices, opType, p2pRank, isSendRecvSelf);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalP2P;
    coalescedDevices_.push_back(devices);
    coalescedComms_.push_back(ncclComms);
  }

  // First let NCCL streams wait for input tensors allocation streams
  syncStreams(devices, ncclEvents_[key], ncclStreams_[key]);

  // Work itself will create the CUDA events on all GPUs of tensors
  bool can_profile = tensors.size() == 1;
  auto work = initWork(
      devices,
      rank_,
      opType,
      can_profile ? profilingTitle : nullptr,
      tensors,
      {});

  // Store references to outputs to be used by WorkNCCL::result and operator<<.
  // Note that these outputs are only valid for recv(), as send() does not
  // modify the inputs but we still create these outputs for use cases such as
  // profiling.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(tensors);

  at::cuda::OptionalCUDAGuard gpuGuard;

  // Start event should only be recorded before the ncclGroupStart()
  if (work->timingEnabled_) {
    for (const auto i : c10::irange(tensors.size())) {
      at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
      (*work->ncclStartEvents_)[i].record(ncclStream);
    }
  }

  pre(ncclStreams_[key], work);

  for (const auto i : c10::irange(tensors.size())) {
    gpuGuard.set_index(devices[i].index());
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];

    // Both send tensor and recv tensor are created on a worker stream and used
    // in different ncclStreams.  Hence, both must record the ncclStream to
    // prevent being freed before the collective finishes.
    //
    // See [Sync Streams].
    c10::cuda::CUDACachingAllocator::recordStream(
        tensors[i].storage().data_ptr(), ncclStream);
  }

  std::vector<void*> comms_;
  if (nccl_use_nonblocking()) {
    for (const auto i : c10::irange(tensors.size())) {
      comms_.push_back((void*)ncclComms[i]->getNcclComm());
    }
  }
  {
    torch::cuda::nccl::AutoNcclGroup nccl_group_guard(
        comms_, nccl_use_nonblocking());
    for (const auto i : c10::irange(tensors.size())) {
      gpuGuard.set_index(devices[i].index());
      at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
#ifndef NCCL_HAS_COMM_NONBLOCKING
      C10D_NCCL_CHECK(
          fn(tensors[i],
             ncclComms[i]->getNcclComm(),
             ncclStream,
             p2pTargetRank),
          ncclComms[i]->getNcclCommFailureReason());
#else
      C10D_NCCL_CHECK_TIMEOUT(
          fn(tensors[i],
             ncclComms[i]->getNcclComm(),
             ncclStream,
             p2pTargetRank),
          ncclComms[i]->getNcclComm(),
          ncclComms[i]->getNcclCommFailureReason());
#endif
    }
  }

  post(ncclStreams_[key]);

  // End event should only be recorded after the ncclGroupEnd()
  for (const auto i : c10::irange(tensors.size())) {
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
    if (!coalescing_state_) {
      (*work->ncclEndEvents_)[i].record(ncclStream);
    }
    work->ncclComms_[i] = ncclComms[i];
    work->blockingWait_ = blockingWait_;
    work->opTimeout_ = options_->timeout;
    work->store_ = store_;
  }

  // Record size info for debug. We only record the size on the first device as
  // multi-device per process is deprecated
  work->numelIn_ = work->numelOut_ = tensors[0].numel();

  // Future only needs to be created and marked completed with outputs for
  // recv(), but still create future for use cases such as profiling even for
  // send().
  {
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStreams_[key]);
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Add a callback that runs profiling end callbacks. wrapCallback() in CUDA
  // future blocks the stream this callback runs on the corresponding
  // ncclEndEvents_ ensuring appropriate synchronization.
  if (work->recordFunctionEndCallback_) {
    work->future_->addCallback(
        [work](at::ivalue::Future& /* unused */) {
          work->recordFunctionEndCallback_();
        },
        // uses_future = false allows us to skip synchronization in
        // ivalue::Future, but is only valid as long as the lambda doesn't use
        // the "Future" argument.
        /*uses_future=*/false);
  }

  // Enqueue P2P op so that it can be cancelled by NCCL watchdog
  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();

  // Notify graphs before we check the capture status preemptively
  at::cuda::CUDAGraph::inc_pending_event_queries();

  if (!coalescing_state_ && capture_status == c10::cuda::CaptureStatus::None) {
    workEnqueue(work);
  } else {
    at::cuda::CUDAGraph::dec_pending_event_queries();
  }

  return work;
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupNCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    OpType opType,
    const char* profilingTitle,
    bool avoidRecordStreams) {
  return collective(
      inputs,
      outputs,
      fn,
      [](std::vector<at::cuda::CUDAStream>&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      [](std::vector<at::cuda::CUDAStream>&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      opType,
      profilingTitle,
      avoidRecordStreams);
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupNCCL::pointToPoint(
    std::vector<at::Tensor>& tensor,
    Fn fn,
    int peer,
    OpType opType,
    const char* profilingTitle) {
  return pointToPoint(
      tensor,
      fn,
      peer,
      opType,
      [](std::vector<at::cuda::CUDAStream>&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      [](std::vector<at::cuda::CUDAStream>&) {},
      profilingTitle);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_sparse(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
#ifdef IS_NCCL_EXP
  std::vector<at::Tensor> outputTensors(tensors.size());
  for (std::vector<at::Tensor>::size_type i = 0; i < tensors.size(); i++) {
    tensors[i] = tensors[i].coalesce();
    outputTensors[i] = torch::zeros(
        tensors[i].sizes(), tensors[i].options().layout(torch::kStrided));
  }
  int dev_in_group = 0;
  auto work = collective(
      tensors,
      outputTensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp = getNcclReduceOp(
            opts.reduceOp, input, ncclDataType, comm, dev_in_group++);

        size_t num_elements = output.numel();
        auto indices = input.indices();
        auto sizes = input.sizes();
        int colSize = sizes[1];
        auto rows = indices[0];
        size_t blockCount = rows.sizes()[0];
        auto recvIndices = indices[0] * colSize;

        // prevent output and recvIndices from being freed
        c10::cuda::CUDACachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        c10::cuda::CUDACachingAllocator::recordStream(
            recvIndices.storage().data_ptr(), stream);
        auto result = ncclAllReduceSparseBlock(
            input._values().data_ptr(), // sendbuff
            recvIndices.data_ptr<int64_t>(), // recv_indices
            blockCount, // block_count
            colSize, // block_length
            output.data_ptr(), // recvbuff
            output.numel(), // recv_count
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
        return result;
      },
      [](std::vector<at::cuda::CUDAStream>& ncclStreams,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      [&](std::vector<at::cuda::CUDAStream>& ncclStreams,
          c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
        // Convert output tensors to sparse and back into tensors.
        for (const auto i : c10::irange(outputTensors.size())) {
          at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
          if (opts.sparseIndices.has_value()) {
            tensors[i] = at::sparse_coo_tensor(
                opts.sparseIndices.value(),
                outputTensors[i],
                tensors[i].sizes());
          } else {
            tensors[i] = outputTensors[i].to_sparse();
          }
        }
      },
      OpType::_ALLREDUCE_SPARSE,
      "nccl:all_reduce_sparse");
  return work;
#else
  // If the nccl branch is not "exp" then we just error
  C10_THROW_ERROR(
      Error,
      "allreduce_sparse is only available in the NCCL experimental branch.");
#endif
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_impl(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  int dev_in_group = 0;
  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp = getNcclReduceOp(
            opts.reduceOp, input, ncclDataType, comm, dev_in_group++);
        return ncclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      OpType::ALLREDUCE,
      "nccl:all_reduce");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  check_gpu_tensors_different_devices(tensors);

  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return allreduce_impl(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  auto total_numel = check_gpu_tensors_same_device(tensors);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce_coalesced", // colName
      total_numel, // inSize
      total_numel, // outSize
      tensors[0].scalar_type(), // dType
      // I'm not sure what in,outSplitSizes mean here.
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  return allreduce_impl(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  check_gpu_tensors_different_devices(tensors);

  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "broadcast", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash tensors.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
        return ncclBcast(
            input.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            root,
            comm,
            stream.stream());
      },
      OpType::BROADCAST,
      "nccl:broadcast",
      avoidRecordStreams);
}

// _broadcast_oop adds an out-of-place broadcast in PGNCCL
// Custom collectives may be implemented by coalescing broadcast operations
// One use-case is implementing a vector all_gather (all_gather_v)
// where unevenly sized inputs are gathered among participating ranks
// Since all_gather provides an out-of-place API, an all_gather_v
// semantic implemented inside pg_nccl.all_gather also needs to support
// out-of-place, for which an out-of-place broadcast is required to be added
c10::intrusive_ptr<Work> ProcessGroupNCCL::_broadcast_oop(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const BroadcastOptions& opts) {
  check_gpu_tensors_different_devices(outputTensors);
  check_gpu_tensors_different_devices(inputTensors);

  // @lint-ignore CLANGTIDY
  auto tensor = outputTensors.back();
  // @lint-ignore CLANGTIDY
  auto in_tensor = inputTensors.back();
  if (tensor.numel() != in_tensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _broadcast_oop must have the same number of elements ");
  }
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() +
          1), // seq + 1 to match collective increment.
      this->getID(),
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "_broadcast_oop", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      this->getSize()); // worldSize

  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank * inputTensors.size() + opts.rootTensor;
        return ncclBroadcast(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            root,
            comm,
            stream.stream());
      },
      OpType::BROADCAST,
      "nccl:_broadcast_oop");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  check_gpu_tensors_different_devices(tensors);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "reduce", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      this->getSize()); // worldSize

  int dev_in_group = 0;
  // avoidRecordStreams_ note: collective() will stash tensors.
  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp = getNcclReduceOp(
            opts.reduceOp, input, ncclDataType, comm, dev_in_group++);
        return ncclReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            root,
            comm,
            stream.stream());
      },
      OpType::REDUCE,
      "nccl:reduce");
}

// _reduce_oop exposes an out-of-place reduce from PGNCCL
// Custom collectives may be implemented by coalescing reduce operations
// One use-case is implementing a vector reduce_scatter (reduce_scatter_v)
// where inputs are reduced and scattered unevenly among participating ranks
// Since reduce_scatter provides an out-of-place API, a reduce_scatter_v
// semantic implemented inside pg_nccl.reduce_scatter also needs to support
// out-of-place, for which an out-of-place reduce is required to be added
c10::intrusive_ptr<Work> ProcessGroupNCCL::_reduce_oop(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ReduceOptions& opts) {
  check_gpu_tensors_different_devices(outputTensors);
  check_gpu_tensors_different_devices(inputTensors);
  // @lint-ignore CLANGTIDY
  auto tensor = outputTensors.back();
  // @lint-ignore CLANGTIDY
  auto in_tensor = inputTensors.back();
  if (tensor.numel() != in_tensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _reduce_oop must have the same number of elements ");
  }
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "_reduce_oop", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      this->getSize()); // worldSize

  int dev_in_group{0};
  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank * inputTensors.size() + opts.rootTensor;
        const auto ncclDataType = getNcclDataType(input.scalar_type());
        const auto ncclReduceOp = getNcclReduceOp(
            opts.reduceOp, input, ncclDataType, comm, dev_in_group++);
        return ncclReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            ncclDataType,
            ncclReduceOp,
            (int)root,
            comm,
            stream.stream());
      },
      OpType::REDUCE,
      "nccl:_reduce_oop");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  check_gpu_tensors_different_devices(inputTensors);
  // @lint-ignore CLANGTIDY
  bool same_size = check_same_size(outputTensors.back());

  if (same_size) {
    auto outputFlattened =
        flatten_for_scatter_gather(outputTensors, inputTensors, size_);
    check_gpu_tensors_different_devices(outputFlattened);

    // @lint-ignore CLANGTIDY
    auto tensor = inputTensors.back();
    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        this->getID(),
        inputTensors, // inputTensors
        outputTensors, // outputTensors
        rank_, // rank
        "all_gather", // colName
        tensor.numel(), // inSize
        tensor.numel() * // outSize
            this->getSize(),
        tensor.scalar_type(), // dType
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>(), // outSplitSize
        this->getSize()); // worldSize

    return collective(
        inputTensors,
        outputFlattened,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          return ncclAllGather(
              input.data_ptr(),
              output.data_ptr(),
              input.numel(),
              getNcclDataType(input.scalar_type()),
              comm,
              stream.stream());
        },
        [](std::vector<at::cuda::CUDAStream>& ncclStreams,
           c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          // avoidRecordStreams_ note: We actually don't need to stash anything
          // here.
          //  - inputTensors is stashed onto work->stashed_for_allocator_safety_
          //    in collective().
          //  - outputFlattened is stashed onto work->outputs_ in collective().
          //  - User-facing outputTensors should be held by the user until after
          //    waiting on work_, or the call makes no sense.
          // So all participating tensors are accounted for, and won't be
          // released back to their allocation streams until after work_ is
          // waited on.
        },
        [&](std::vector<at::cuda::CUDAStream>& ncclStreams,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          // Copy the flattened output tensors to the outputs.
          for (const auto i : c10::irange(outputTensors.size())) {
            at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
            for (const auto j : c10::irange(outputTensors[0].size())) {
              // See [Sync Streams].
              if (!avoidRecordStreams_) {
                c10::cuda::CUDACachingAllocator::recordStream(
                    outputTensors[i][j].storage().data_ptr(), ncclStreams[i]);
              }
              outputTensors[i][j].copy_(outputFlattened[i][j], true);
            }
          }
        },
        OpType::ALLGATHER,
        "nccl:all_gather");
  } else {
    const auto num_devices = outputTensors.size();
    const auto num_reduces = outputTensors[0].size();
    std::vector<c10::intrusive_ptr<Work>> works;
    startCoalescing();
    for (const auto i : c10::irange(num_reduces)) {
      std::vector<at::Tensor> inputs_multi_dev(num_devices);
      std::vector<at::Tensor> outputs_multi_dev(num_devices);
      for (const auto j : c10::irange(num_devices)) {
        // @lint-ignore CLANGTIDY
        outputs_multi_dev[j] = outputTensors[j][i];
        inputs_multi_dev[j] =
            // @lint-ignore CLANGTIDY
            i == (rank_ * num_devices + j) ? inputTensors[j]
                                           : outputs_multi_dev[j];
      }
      auto broadcastOpts = BroadcastOptions{
          static_cast<int64_t>(i / num_devices),
          static_cast<int64_t>(i % num_devices),
          opts.timeout};
      auto work =
          _broadcast_oop(outputs_multi_dev, inputs_multi_dev, broadcastOpts);
      works.push_back(work);
    }
    auto work = endCoalescing();
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError,
      "ProcessGroupNCCL does not support allgather_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        return ncclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      OpType::COALESCED,
      "nccl:all_gather_into_tensor_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  check_gpu_tensors_different_devices(outputTensors);
  // @lint-ignore CLANGTIDY
  bool same_size = check_same_size(inputTensors.back());

  if (same_size) {
    // @lint-ignore CLANGTIDY
    auto tensor = outputTensors.back();

    int dev_in_group{0};
    auto inputFlattened =
        flatten_for_scatter_gather(inputTensors, outputTensors, size_);
    check_gpu_tensors_different_devices(inputFlattened);

    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        this->getID(),
        inputTensors, // inputTensors
        outputTensors, // outputTensors
        rank_, // rank
        "reduce_scatter", // colName
        tensor.numel() * this->getSize(), // inSize
        tensor.numel(), // outSize
        tensor.scalar_type(), // dType
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>(), // outSplitSizes
        this->getSize()); // worldSize

    return collective(
        inputFlattened,
        outputTensors,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          const auto ncclDataType = getNcclDataType(input.scalar_type());
          const auto ncclReduceOp = getNcclReduceOp(
              opts.reduceOp, input, ncclDataType, comm, dev_in_group++);
          return ncclReduceScatter(
              input.data_ptr(),
              output.data_ptr(),
              output.numel(),
              ncclDataType,
              ncclReduceOp,
              comm,
              stream.stream());
        },
        [&](std::vector<at::cuda::CUDAStream>& ncclStreams,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
          if (avoidRecordStreams_) {
            // We only need to stash inputTensors.
            //  - inputFlattened is stashed onto
            //  work->stashed_for_allocator_safety_
            //    in collective().
            //  - User-facing outputTensors is stashed onto work->outputs_ in
            //  collective(),
            //    and should also be held by the user until after waiting on
            //    work_.
            auto& v = work->stashed_for_allocator_safety_;
            for (const auto i : c10::irange(inputTensors.size())) {
              v->insert(
                  v->end(), inputTensors[i].begin(), inputTensors[i].end());
            }
          }

          // Copy the input tensors to the flattened inputs.
          for (const auto i : c10::irange(inputTensors.size())) {
            at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
            for (const auto j : c10::irange(inputTensors[0].size())) {
              // See [Sync Streams].
              if (!avoidRecordStreams_) {
                c10::cuda::CUDACachingAllocator::recordStream(
                    inputTensors[i][j].storage().data_ptr(), ncclStreams[i]);
              }
              inputFlattened[i][j].copy_(inputTensors[i][j], true);
            }
          }
        },
        [&](std::vector<at::cuda::CUDAStream>&,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
        OpType::REDUCE_SCATTER,
        "nccl:reduce_scatter");
  } else {
    const auto num_devices = inputTensors.size();
    const auto num_reduces = inputTensors[0].size();
    std::vector<c10::intrusive_ptr<Work>> works;
    startCoalescing();
    for (const auto i : c10::irange(num_reduces)) {
      std::vector<at::Tensor> inputs_multi_dev(num_devices);
      std::vector<at::Tensor> outputs_multi_dev(num_devices);
      for (const auto j : c10::irange(num_devices)) {
        // @lint-ignore CLANGTIDY
        inputs_multi_dev[j] = inputTensors[j][i];
        outputs_multi_dev[j] =
            // @lint-ignore CLANGTIDY
            i == (rank_ * num_devices + j) ? outputTensors[j]
                                           : inputs_multi_dev[j];
      }
      auto reduceOpts = ReduceOptions{
          opts.reduceOp,
          static_cast<int64_t>(i / num_devices),
          static_cast<int64_t>(i % num_devices),
          opts.timeout};
      auto work = _reduce_oop(outputs_multi_dev, inputs_multi_dev, reduceOpts);
      works.push_back(work);
    }
    auto work = endCoalescing();
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  if (inputTensor.dtype() != outputTensor.dtype()) {
    C10_THROW_ERROR(
        TypeError, "input tensor must be the same type as the output tensor.");
  }

  if (inputTensor.numel() != outputTensor.numel() * size_) {
    C10_THROW_ERROR(
        ValueError,
        "input tensor must be the same size as output size times world size");
  }

  // @lint-ignore CLANGTIDY
  const auto& tensor = outputTensor;
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      inputTensor, // inputTensor
      outputTensor, // outputTensor
      rank_, // rank
      "_reduce_scatter_base", // colName
      inputTensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dtype
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      this->getSize()); // worldSize

  auto inputs = std::vector<at::Tensor>{inputTensor};
  auto outputs = std::vector<at::Tensor>{outputTensor};

  int dev_in_group = 0;
  // avoidRecordStreams_ note: collective() will stash inputs and outputs.
  // Note 2: for asyncOp = false, we don't want to record streams because we
  // know that the NCCL stream will join back to the "current" stream right
  // after this op. So we might just as well keep the stream ownership of the
  // input/output tensors unchanged. The benefit would be that the
  // allocation/free of the tensors would look deterministic to the "current"
  // stream so that the caching allocator can reuse memory pool for this stream
  // in a clever way. This setting is added for libraries like FSDP which uses
  // `reduce_scatter_tensor`.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        if (!avoidRecordStreams) {
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp = getNcclReduceOp(
            opts.reduceOp, input, ncclDataType, comm, dev_in_group++);
        return ncclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            output.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      OpType::_REDUCE_SCATTER_BASE,
      "nccl:_reduce_scatter_base",
      avoidRecordStreams);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        if (!avoidRecordStreams_) {
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        auto ncclDataType = getNcclDataType(input.scalar_type());
        auto ncclReduceOp = getNcclReduceOp(
            opts.reduceOp, input, ncclDataType, comm, /*dev_in_group=*/0);
        return ncclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            output.numel(),
            ncclDataType,
            ncclReduceOp,
            comm,
            stream.stream());
      },
      OpType::COALESCED,
      "nccl:reduce_scatter_tensor_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::barrier(const BarrierOptions& opts) {
  RECORD_PARAM_COMMS(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      rank_, // rank
      "barrier", // colName
      0, // inSize
      0, // outSize
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      this->getSize()); // worldSize

  std::vector<at::Device> devices;

  // Use user defined GPU device ids if provided
  if (!opts.device_ids.empty()) {
    for (auto device : opts.device_ids) {
      devices.emplace_back(at::DeviceType::CUDA, device);
    }
  } else if (usedDeviceIdxs_.empty()) {
    // This means there is not yet a NCCL collective being called
    // Here we have to use the best guesses and will use a single GPU to call
    // allreduce to achieve barrier.
    // In case the multiple processes fall into the same node, we use rank to
    // ensure that each process is on a different GPU
    auto numGPUs = at::cuda::getNumGPUs();
    int16_t deviceIdx = static_cast<int16_t>(rank_ % numGPUs);
    LOG(INFO) << c10::str(
        "Rank ",
        this->getRank(),
        " using GPU ",
        deviceIdx,
        " to perform barrier as devices used by this process are currently unknown. ",
        "This can potentially cause a hang if this rank to GPU mapping is incorrect.",
        "Specify device_ids in barrier() to force use of a particular device.");
    devices.emplace_back(getDeviceForRank(rank_));
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.emplace_back(at::DeviceType::CUDA, usedDeviceIdx);
    }
  }

  std::vector<at::Tensor> barrierTensors;
  barrierTensors.reserve(devices.size());

  at::cuda::OptionalCUDAGuard gpuGuard;
  for (auto& device : devices) {
    gpuGuard.set_index(device.index());
    barrierTensors.push_back(at::empty(
        {1},
        at::TensorOptions().device(at::DeviceType::CUDA).dtype(at::kByte)));
  }

  // All reduce to achieve the barrier
  auto work = allreduce(barrierTensors);

  // Work will take over barrierTensors
  auto ncclWork = dynamic_cast<ProcessGroupNCCL::WorkNCCL*>(work.get());
  TORCH_CHECK(ncclWork);
  ncclWork->barrierTensors_ = std::move(barrierTensors);

  return work;
}

#ifdef ENABLE_NCCL_P2P_SUPPORT
c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  check_gpu_single_tensor(outputTensor);
  check_gpu_single_tensor(inputTensor);
  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};

    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        this->getID(),
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_all", // colName
        inputTensor.numel(), // inSize
        outputTensor.numel(), // outSize
        inputTensor.scalar_type(), // dType
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>(), // outSplitSizes
        this->getSize()); // worldSize

    // avoidRecordStreams_ note: collective() will stash inputTensors and
    // outputTensors.
    return collective(
        inputTensors,
        outputTensors,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          // See [Sync Streams].
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          torch::cuda::nccl::all2all_single_equal_split(
              input, output, this->getSize(), comm, stream);
          return ncclSuccess;
        },
        OpType::ALLTOALL_BASE,
        "nccl:all_to_all");
  } else {
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};

    RECORD_PARAM_COMMS_DATA(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        this->getID(),
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_allv", // colName
        inputTensor.numel(), // inSize
        outputTensor.numel(), // outSize
        inputTensor.scalar_type(), // dType
        inputSplitSizes, // inSplitSizes
        outputSplitSizes, // outSplitSizes
        this->getSize()); // worldSize

    // avoidRecordStreams_ note: collective() will stash inputTensors and
    // outputTensors.
    return collective(
        inputTensors,
        outputTensors,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          std::vector<size_t> send_lengths(size_);
          std::vector<size_t> recv_lengths(size_);
          std::vector<size_t> send_offsets(size_);
          std::vector<size_t> recv_offsets(size_);
          c10d::computeLengthsAndOffsets(
              inputSplitSizes, input, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              outputSplitSizes, output, &recv_lengths, &recv_offsets);
          // See [Sync Streams].
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          torch::cuda::nccl::all2all_single_unequal_split(
              input.data_ptr(),
              send_lengths.data(),
              send_offsets.data(),
              output.data_ptr(),
              recv_lengths.data(),
              recv_offsets.data(),
              input.element_size(),
              input.scalar_type(),
              comm,
              stream);
          return ncclSuccess;
        },
        OpType::ALLTOALL_BASE,
        "nccl:all_to_all");
  }
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& /* unused */) {
  std::vector<int64_t> inSplitSizes;
  std::vector<int64_t> outSplitSizes;
  int64_t total_numel = 0;

  auto device = outputTensors[0].device();
  for (const auto r : c10::irange(outputTensors.size())) {
    check_gpu_single_tensor(outputTensors[r]);
    check_gpu_single_tensor(inputTensors[r]);
    TORCH_CHECK(
        device == outputTensors[r].device() &&
            device == inputTensors[r].device(),
        "Tensors must be on the same device")
    inSplitSizes.push_back(inputTensors[r].numel());
    outSplitSizes.push_back(outputTensors[r].numel());
    total_numel += inputTensors[r].numel();
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "all_to_all", // colName
      total_numel, // inSize
      total_numel, // outSize
      inputTensors.front().scalar_type(), // dType
      inSplitSizes, // inSplitSizes
      outSplitSizes, // outSplitSizes
      this->getSize()); // worldSize

  std::vector<at::Tensor> inputTensor0 = {inputTensors[0]};
  std::vector<at::Tensor> outputTensor0 = {outputTensors[0]};
  return collective(
      inputTensor0,
      outputTensor0,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        torch::cuda::nccl::all2all(outputTensors, inputTensors, comm, stream);
        return ncclSuccess;
      },
      [&](std::vector<at::cuda::CUDAStream>&,
          c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {
        if (avoidRecordStreams_) {
          // inputTensor0 and outputTensor0 are stashed redundantly by
          // collective(), but that's ok.
          auto& v = work->stashed_for_allocator_safety_;
          v->insert(v->end(), inputTensors.begin(), inputTensors.end());
          v->insert(v->end(), outputTensors.begin(), outputTensors.end());
        }
      },
      [](std::vector<at::cuda::CUDAStream>&,
         c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
      OpType::ALLTOALL,
      "nccl:all_to_all");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /* unused */) {
  check_gpu_tensors_different_devices(tensors);

  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      tensors, // inputTensors
      tensors, // outputTensors
      dstRank, // rank
      "send", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      this->getSize()); // worldSize

  auto ret = pointToPoint(
      tensors,
      [&](at::Tensor& input,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream,
          int dst) {
        torch::cuda::nccl::send(input, comm, stream, dst);
        return ncclSuccess;
      },
      dstRank,
      OpType::SEND,
      c10::str("nccl:send ", rank_, "->", dstRank).c_str());
  return ret;
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /* unused */) {
  check_gpu_tensors_different_devices(tensors);

  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      tensors, // inputTensors
      tensors, // outputTensors
      srcRank, // rank
      "recv", // colName
      tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      this->getSize()); // worldSize

  auto ret = pointToPoint(
      tensors,
      [&](at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream,
          int src) {
        torch::cuda::nccl::recv(output, comm, stream, src);
        return ncclSuccess;
      },
      srcRank,
      OpType::RECV,
      c10::str("nccl:recv ", rank_, "<-", srcRank).c_str());
  return ret;
}
#else
c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    std::vector<int64_t>& /* unused */,
    std::vector<int64_t>& /* unused */,
    const AllToAllOptions& /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError,
      "ProcessGroupNCCL only supports alltoall* for NCCL lib version >= 2.7.0");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError,
      "ProcessGroupNCCL only supports alltoall* for NCCL lib version >= 2.7.0");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError,
      "ProcessGroupNCCL only supports send for NCCL lib version >= 2.7.0");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError,
      "ProcessGroupNCCL only supports recv for NCCL lib version >= 2.7.0");
}
#endif

void ProcessGroupNCCL::groupStart() {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
  C10D_NCCL_CHECK(ncclGroupStart(), c10::nullopt);
#endif
  ++ncclActiveGroupCounter_;
}

void ProcessGroupNCCL::groupEnd() {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
#else
  if (!nccl_use_nonblocking()) {
    C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
  } else {
    TORCH_WARN(
        "ProcessGroupNCCL::groupEnd() called in nonblocking communicator mode without involved communicators specified; gathering all mapped communicators...");
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<NCCLComm>> ncclComms_;
    for (auto& it : devNCCLCommMap_) {
      ncclComms_.insert(ncclComms_.end(), it.second.begin(), it.second.end());
    }
    C10D_NCCL_CHECK_TIMEOUT_GROUPEND(ncclGroupEnd(), ncclComms_, c10::nullopt);
  }
#endif
#endif
  --ncclActiveGroupCounter_;
}

void ProcessGroupNCCL::groupEndNonblocking(
    std::vector<std::shared_ptr<NCCLComm>> comms) {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
#else
  if (!nccl_use_nonblocking()) {
    C10D_NCCL_CHECK(ncclGroupEnd(), c10::nullopt);
  } else {
    C10D_NCCL_CHECK_TIMEOUT_GROUPEND(ncclGroupEnd(), comms, c10::nullopt);
  }
#endif
#endif
  --ncclActiveGroupCounter_;
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupNCCL::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  check_gpu_tensors_different_devices(inputTensors);
  assertSingleElementInput(invalidArgument, inputTensors);

  // @lint-ignore CLANGTIDY
  auto tensor = inputTensors.back();

  std::vector<at::Tensor> outputs;

  if (getRank() == opts.rootRank) {
    if (outputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (outputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect output list size " << outputTensors[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = inputTensors[0].options();
    const auto& sizes = inputTensors[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, outputTensors[0], options, sizes);
    outputs = outputTensors[0];
  } else {
    // if not in the root rank, initialize outputs as empty list
    if (outputTensors.size() != 0) {
      invalidArgument("requires empty output on non-root");
    }
    outputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    outputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "gather", // colName
      tensor.numel(), // inSize
      tensor.numel() * this->getSize(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash inputTensors and
  // outputs, which == outputTensors[0] on the root rank where it matters.
  return collective(
      inputTensors,
      outputs,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoidRecordStreams_) {
            for (auto output : outputs) {
              c10::cuda::CUDACachingAllocator::recordStream(
                  output.storage().data_ptr(), stream);
            }
          }
        }
        torch::cuda::nccl::gather(inputTensors[0], outputs, comm, stream, root);
        return ncclSuccess;
      },
      OpType::GATHER,
      "nccl:gather");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupNCCL::scatter: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  check_gpu_tensors_different_devices(outputTensors);
  assertSingleElementInput(invalidArgument, outputTensors);

  // @lint-ignore CLANGTIDY
  auto tensor = outputTensors.back();

  std::vector<at::Tensor> inputs;

  if (getRank() == opts.rootRank) {
    if (inputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (inputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect input list size " << inputTensors[0].size()
         << ". Input list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = outputTensors[0].options();
    const auto& sizes = outputTensors[0].sizes();
    assertTypeAndSizesMatch(invalidArgument, inputTensors[0], options, sizes);
    inputs = inputTensors[0];
  } else {
    // if not in the root rank, initialize inputTensors as empty place holder
    // with an empty list
    if (inputTensors.size() != 0) {
      invalidArgument("requires empty input on non-root");
    }
    inputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    inputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "scatter", // colName
      tensor.numel(), // inSize
      tensor.numel() * this->getSize(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      this->getSize()); // worldSize

  // avoidRecordStreams_ note: collective() will stash outputTensors and
  // inputs, which == inputTensors[0] on the root rank where it matters.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      outputTensors,
      inputs,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoidRecordStreams) {
            for (auto input : inputs) {
              c10::cuda::CUDACachingAllocator::recordStream(
                  input.storage().data_ptr(), stream);
            }
          }
        }
        torch::cuda::nccl::scatter(
            inputs, outputTensors[0], comm, stream, root);
        return ncclSuccess;
      },
      OpType::SCATTER,
      "nccl:scatter",
      avoidRecordStreams);
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  C10_THROW_ERROR(
      NotImplementedError, "ProcessGroupNCCL does not support recvAnysource");
}

c10::intrusive_ptr<Work> ProcessGroupNCCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  check_gpu_single_tensor(input_tensor);
  check_gpu_single_tensor(output_tensor);

  if (input_tensor.dtype() != output_tensor.dtype()) {
    C10_THROW_ERROR(
        TypeError, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "output tensor size must be equal to world_size times input tensor size");
  }

  // @lint-ignore CLANGTIDY
  const auto& tensor = output_tensor;
  RECORD_PARAM_COMMS_DATA(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      this->getID(),
      input_tensor, // inputTensors
      output_tensor, // outputTensors
      rank_, // rank
      "_allgather_base", // colName
      input_tensor.numel(), // inSize
      tensor.numel(), // outSize
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      this->getSize()); // worldSize

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor>{input_tensor};
  auto outputs = std::vector<at::Tensor>{output_tensor};

  // avoidRecordStreams_ note: collective() will stash inputs and outputs.
  // Note 2: for asyncOp = false, we don't want to record streams because we
  // know that the NCCL stream will join back to the "current" stream right
  // after this op. So we might just as well keep the stream ownership of the
  // input/output tensors unchanged. The benefit would be that the
  // allocation/free of the tensors would look deterministic to the "current"
  // stream so that the caching allocator can reuse memory pool for this stream
  // in a clever way. This setting is added for libraries like FSDP which uses
  // `all_gather_into_tensor`.
  bool avoidRecordStreams = avoidRecordStreams_ || (!opts.asyncOp);

  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream) {
        if (!avoidRecordStreams) {
          c10::cuda::CUDACachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        return ncclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getNcclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      OpType::_ALLGATHER_BASE,
      "nccl:_all_gather_base",
      avoidRecordStreams);
}

#ifdef USE_NCCL_WITH_UCC
std::shared_ptr<at::DynamicLibrary> ProcessGroupNCCL::uccLib_ = nullptr;
#endif

bool ProcessGroupNCCL::isUCCAvailable() const {
#ifdef USE_NCCL_WITH_UCC
  return (uccPG_ != nullptr);
#else
  return false;
#endif
}

} // namespace c10d

#endif // USE_C10D_NCCL
