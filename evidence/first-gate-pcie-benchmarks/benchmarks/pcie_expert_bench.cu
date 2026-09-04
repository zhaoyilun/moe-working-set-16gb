#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kExpertBytes = 2'764'800;
constexpr int kHidden = 2560;
constexpr int kIntermediate = 640;
constexpr int kTopK = 10;
constexpr int kLayers = 48;
constexpr int kExpertsPerLayer = 512;
constexpr std::size_t kPieceBytes[9] = {
    819'200, 51'200, 51'200,
    819'200, 51'200, 51'200,
    819'200, 51'200, 51'200,
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream out;
        out << file << ':' << line << " CUDA failure in " << expression << ": "
            << cudaGetErrorString(status);
        fail(out.str());
    }
}

void checkCublas(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream out;
        out << file << ':' << line << " cuBLAS failure in " << expression
            << ": status=" << static_cast<int>(status);
        fail(out.str());
    }
}

#define CUDA_CHECK(expr) checkCuda((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) checkCublas((expr), #expr, __FILE__, __LINE__)

struct Options {
    std::vector<double> poolGB{1.0, 2.0, 4.0, 6.0, 8.0};
    std::vector<int> batches{1, 2, 4, 8, 10, 20};
    double targetCopyGB = 1.0;
    int minIterations = 40;
    int warmup = 5;
    int pipelineCycles = 80;
    int pipelineRounds = 9;
    std::filesystem::path outputDir = "results/latest";
};

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> result;
    std::stringstream input(value);
    std::string part;
    while (std::getline(input, part, delimiter)) {
        if (!part.empty()) result.push_back(part);
    }
    return result;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) fail("missing value after " + arg);
            return argv[i];
        };
        if (arg == "--pool-gb") {
            options.poolGB.clear();
            for (const auto& s : split(next(), ',')) options.poolGB.push_back(std::stod(s));
        } else if (arg == "--batches") {
            options.batches.clear();
            for (const auto& s : split(next(), ',')) options.batches.push_back(std::stoi(s));
        } else if (arg == "--target-copy-gb") {
            options.targetCopyGB = std::stod(next());
        } else if (arg == "--min-iterations") {
            options.minIterations = std::stoi(next());
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(next());
        } else if (arg == "--pipeline-cycles") {
            options.pipelineCycles = std::stoi(next());
        } else if (arg == "--pipeline-rounds") {
            options.pipelineRounds = std::stoi(next());
        } else if (arg == "--output-dir") {
            options.outputDir = next();
        } else if (arg == "--help") {
            std::cout
                << "slotstream_bench options:\n"
                << "  --pool-gb 1,2,4,6,8\n"
                << "  --batches 1,2,4,8,10,20\n"
                << "  --target-copy-gb 1\n"
                << "  --min-iterations 40\n"
                << "  --warmup 5\n"
                << "  --pipeline-cycles 80\n"
                << "  --pipeline-rounds 9\n"
                << "  --output-dir PATH\n";
            std::exit(0);
        } else {
            fail("unknown option: " + arg);
        }
    }
    if (options.poolGB.empty() || options.batches.empty()) fail("pool and batch lists must be non-empty");
    if (options.targetCopyGB <= 0 || options.minIterations < 1 || options.warmup < 0 ||
        options.pipelineCycles < 2 || options.pipelineRounds < 3) {
        fail("invalid non-positive benchmark option");
    }
    for (double gb : options.poolGB) if (gb <= 0) fail("pool sizes must be positive");
    for (int n : options.batches) if (n <= 0) fail("batch sizes must be positive");
    return options;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(p * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

double mean(const std::vector<double>& values) {
    return values.empty() ? 0.0
                          : std::accumulate(values.begin(), values.end(), 0.0) /
                                static_cast<double>(values.size());
}

struct CopyStats {
    double wallMeanMs = 0;
    double wallP50Ms = 0;
    double wallP95Ms = 0;
    double eventMeanMs = 0;
    double effectiveGBs = 0;
    int iterations = 0;
};

enum class Layout { ContiguousRecord, NinePiecesPerExpert };

const char* layoutName(Layout layout) {
    return layout == Layout::ContiguousRecord ? "contiguous_record" : "nine_pieces_per_expert";
}

void enqueueCopy(Layout layout, std::uint8_t* destination, const std::uint8_t* source,
                 int batchExperts, cudaStream_t stream) {
    if (layout == Layout::ContiguousRecord) {
        CUDA_CHECK(cudaMemcpyAsync(destination, source,
                                   static_cast<std::size_t>(batchExperts) * kExpertBytes,
                                   cudaMemcpyHostToDevice, stream));
        return;
    }
    std::size_t offset = 0;
    for (int expert = 0; expert < batchExperts; ++expert) {
        for (std::size_t piece : kPieceBytes) {
            CUDA_CHECK(cudaMemcpyAsync(destination + offset, source + offset, piece,
                                       cudaMemcpyHostToDevice, stream));
            offset += piece;
        }
    }
}

CopyStats benchmarkCopy(Layout layout, std::uint8_t* devicePool, std::size_t poolBytes,
                        const std::uint8_t* host, int batchExperts,
                        const Options& options, cudaStream_t stream) {
    const std::size_t bytes = static_cast<std::size_t>(batchExperts) * kExpertBytes;
    if (bytes > poolBytes) fail("copy batch is larger than the device pool");
    const std::size_t poolSlots = poolBytes / kExpertBytes;
    const std::size_t usableStarts = poolSlots - static_cast<std::size_t>(batchExperts) + 1;
    const int iterations = std::max(
        options.minIterations,
        static_cast<int>(std::ceil(options.targetCopyGB * 1e9 / static_cast<double>(bytes))));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < options.warmup; ++i) {
        const std::size_t slot = (static_cast<std::size_t>(i) * 7919) % usableStarts;
        enqueueCopy(layout, devicePool + slot * kExpertBytes, host, batchExperts, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<double> wallMs;
    std::vector<double> eventMs;
    wallMs.reserve(iterations);
    eventMs.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        const std::size_t slot = (static_cast<std::size_t>(i + options.warmup) * 7919) % usableStarts;
        CUDA_CHECK(cudaEventRecord(start, stream));
        const auto wallStart = std::chrono::steady_clock::now();
        enqueueCopy(layout, devicePool + slot * kExpertBytes, host, batchExperts, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        const auto wallStop = std::chrono::steady_clock::now();
        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        wallMs.push_back(std::chrono::duration<double, std::milli>(wallStop - wallStart).count());
        eventMs.push_back(elapsed);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CopyStats stats;
    stats.wallMeanMs = mean(wallMs);
    stats.wallP50Ms = percentile(wallMs, 0.50);
    stats.wallP95Ms = percentile(wallMs, 0.95);
    stats.eventMeanMs = mean(eventMs);
    stats.effectiveGBs = static_cast<double>(bytes) / (stats.wallMeanMs / 1000.0) / 1e9;
    stats.iterations = iterations;
    return stats;
}

void validateCopyPath(Layout layout, std::uint8_t* devicePool,
                      const std::uint8_t* pinnedHost, int batchExperts,
                      cudaStream_t stream) {
    const std::size_t bytes = static_cast<std::size_t>(batchExperts) * kExpertBytes;
    std::vector<std::uint8_t> received(bytes);
    enqueueCopy(layout, devicePool, pinnedHost, batchExperts, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(received.data(), devicePool, bytes, cudaMemcpyDeviceToHost));
    if (std::memcmp(received.data(), pinnedHost, bytes) != 0) {
        fail(std::string("copy validation failed for layout ") + layoutName(layout));
    }
}

class EquivalentMoeCompute {
public:
    explicit EquivalentMoeCompute(int maxExperts) : maxExperts_(maxExperts) {
        const std::size_t matrixElements = static_cast<std::size_t>(maxExperts_) * kHidden * kIntermediate;
        const std::size_t inputElements = static_cast<std::size_t>(maxExperts_) * kHidden;
        const std::size_t intermediateElements = static_cast<std::size_t>(maxExperts_) * kIntermediate;
        CUDA_CHECK(cudaMalloc(&gateWeight_, matrixElements * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&upWeight_, matrixElements * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&downWeight_, matrixElements * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&input_, inputElements * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&gate_, intermediateElements * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&up_, intermediateElements * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&output_, inputElements * sizeof(__half)));
        CUDA_CHECK(cudaMemset(gateWeight_, 1, matrixElements * sizeof(__half)));
        CUDA_CHECK(cudaMemset(upWeight_, 2, matrixElements * sizeof(__half)));
        CUDA_CHECK(cudaMemset(downWeight_, 3, matrixElements * sizeof(__half)));
        CUDA_CHECK(cudaMemset(input_, 1, inputElements * sizeof(__half)));
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
    }

    ~EquivalentMoeCompute() {
        if (handle_) cublasDestroy(handle_);
        cudaFree(output_);
        cudaFree(up_);
        cudaFree(gate_);
        cudaFree(input_);
        cudaFree(downWeight_);
        cudaFree(upWeight_);
        cudaFree(gateWeight_);
    }

    EquivalentMoeCompute(const EquivalentMoeCompute&) = delete;
    EquivalentMoeCompute& operator=(const EquivalentMoeCompute&) = delete;

    void enqueue(int experts, cudaStream_t stream) {
        if (experts > maxExperts_) fail("compute batch exceeds allocation");
        CUBLAS_CHECK(cublasSetStream(handle_, stream));
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const long long firstWeightStride = static_cast<long long>(kIntermediate) * kHidden;
        const long long secondWeightStride = static_cast<long long>(kHidden) * kIntermediate;

        auto firstProjection = [&](const __half* weight, __half* output) {
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                kIntermediate, 1, kHidden,
                &alpha,
                weight, CUDA_R_16F, kIntermediate, firstWeightStride,
                input_, CUDA_R_16F, kHidden, kHidden,
                &beta,
                output, CUDA_R_16F, kIntermediate, kIntermediate,
                experts, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };
        firstProjection(gateWeight_, gate_);
        firstProjection(upWeight_, up_);
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            kHidden, 1, kIntermediate,
            &alpha,
            downWeight_, CUDA_R_16F, kHidden, secondWeightStride,
            gate_, CUDA_R_16F, kIntermediate, kIntermediate,
            &beta,
            output_, CUDA_R_16F, kHidden, kHidden,
            experts, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    static double flopsPerExpert() {
        return 3.0 * 2.0 * static_cast<double>(kHidden) * kIntermediate;
    }

private:
    int maxExperts_ = 0;
    cublasHandle_t handle_ = nullptr;
    __half* gateWeight_ = nullptr;
    __half* upWeight_ = nullptr;
    __half* downWeight_ = nullptr;
    __half* input_ = nullptr;
    __half* gate_ = nullptr;
    __half* up_ = nullptr;
    __half* output_ = nullptr;
};

struct PipelineStats {
    double copyOnlyMs = 0;
    double computeOnlyMs = 0;
    double serialMs = 0;
    double doubleBufferedSerialMs = 0;
    double overlapMeanMs = 0;
    double overlapP50Ms = 0;
    double overlapP95Ms = 0;
    double exposedCopyMs = 0;
    double hiddenCopyPercent = 0;
};

double timedRound(const std::function<void()>& enqueue, int cycles) {
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    enqueue();
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           static_cast<double>(cycles);
}

PipelineStats benchmarkPipeline(std::uint8_t* devicePool, std::size_t poolBytes,
                                const std::uint8_t* pinnedHost, int experts,
                                const Options& options, EquivalentMoeCompute& compute) {
    const std::size_t bytes = static_cast<std::size_t>(experts) * kExpertBytes;
    if (2 * bytes > poolBytes) fail("pipeline needs two device buffers inside the pool");
    const int cycles = options.pipelineCycles;

    cudaStream_t serialStream = nullptr;
    cudaStream_t copyStream = nullptr;
    cudaStream_t computeStream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&serialStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));

    std::vector<cudaEvent_t> ready(cycles);
    std::vector<cudaEvent_t> consumed(cycles);
    for (int i = 0; i < cycles; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ready[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&consumed[i], cudaEventDisableTiming));
    }

    auto copyOnly = [&] {
        for (int i = 0; i < cycles; ++i) {
            enqueueCopy(Layout::ContiguousRecord,
                        devicePool + static_cast<std::size_t>(i % 2) * bytes,
                        pinnedHost + static_cast<std::size_t>(i % 2) * bytes,
                        experts, copyStream);
        }
    };
    auto computeOnly = [&] {
        for (int i = 0; i < cycles; ++i) compute.enqueue(experts, computeStream);
    };
    auto serial = [&] {
        for (int i = 0; i < cycles; ++i) {
            enqueueCopy(Layout::ContiguousRecord, devicePool, pinnedHost, experts, serialStream);
            compute.enqueue(experts, serialStream);
        }
    };
    auto doubleBufferedSerial = [&] {
        for (int i = 0; i < cycles; ++i) {
            const std::size_t offset = static_cast<std::size_t>(i % 2) * bytes;
            enqueueCopy(Layout::ContiguousRecord, devicePool + offset, pinnedHost + offset,
                        experts, serialStream);
            compute.enqueue(experts, serialStream);
        }
    };
    auto overlapped = [&] {
        for (int i = 0; i < cycles; ++i) {
            const int buffer = i % 2;
            if (i >= 2) CUDA_CHECK(cudaStreamWaitEvent(copyStream, consumed[i - 2], 0));
            const std::size_t offset = static_cast<std::size_t>(buffer) * bytes;
            enqueueCopy(Layout::ContiguousRecord, devicePool + offset, pinnedHost + offset,
                        experts, copyStream);
            CUDA_CHECK(cudaEventRecord(ready[i], copyStream));
            CUDA_CHECK(cudaStreamWaitEvent(computeStream, ready[i], 0));
            compute.enqueue(experts, computeStream);
            CUDA_CHECK(cudaEventRecord(consumed[i], computeStream));
        }
    };

    copyOnly();
    computeOnly();
    serial();
    doubleBufferedSerial();
    overlapped();
    CUDA_CHECK(cudaDeviceSynchronize());

    PipelineStats result;
    std::vector<double> copySamples;
    std::vector<double> computeSamples;
    std::vector<double> serialSamples;
    std::vector<double> doubleSamples;
    std::vector<double> overlapSamples;
    for (int round = 0; round < options.pipelineRounds; ++round) {
        copySamples.push_back(timedRound(copyOnly, cycles));
        computeSamples.push_back(timedRound(computeOnly, cycles));
        serialSamples.push_back(timedRound(serial, cycles));
        doubleSamples.push_back(timedRound(doubleBufferedSerial, cycles));
        overlapSamples.push_back(timedRound(overlapped, cycles));
    }
    result.copyOnlyMs = mean(copySamples);
    result.computeOnlyMs = mean(computeSamples);
    result.serialMs = mean(serialSamples);
    result.doubleBufferedSerialMs = mean(doubleSamples);
    result.overlapMeanMs = mean(overlapSamples);
    result.overlapP50Ms = percentile(overlapSamples, 0.50);
    result.overlapP95Ms = percentile(overlapSamples, 0.95);
    result.exposedCopyMs = std::max(0.0, result.overlapMeanMs - result.computeOnlyMs);
    result.hiddenCopyPercent = result.copyOnlyMs > 0
        ? 100.0 * std::clamp(1.0 - result.exposedCopyMs / result.copyOnlyMs, 0.0, 1.0)
        : 0.0;

    for (int i = 0; i < cycles; ++i) {
        CUDA_CHECK(cudaEventDestroy(ready[i]));
        CUDA_CHECK(cudaEventDestroy(consumed[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(copyStream));
    CUDA_CHECK(cudaStreamDestroy(serialStream));
    return result;
}

std::string timestampUtc() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        std::filesystem::create_directories(options.outputDir);

        int device = 0;
        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        int driverVersion = 0;
        int runtimeVersion = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

        const int maxBatch = *std::max_element(options.batches.begin(), options.batches.end());
        const std::size_t maxHostBytes = 2ull * static_cast<std::size_t>(maxBatch) * kExpertBytes;
        std::vector<std::uint8_t> pageable(maxHostBytes, 0x5a);
        std::uint8_t* pinned = nullptr;
        CUDA_CHECK(cudaMallocHost(&pinned, maxHostBytes));
        std::memset(pinned, 0xa5, maxHostBytes);

        cudaStream_t copyStream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));

        std::ofstream metadata(options.outputDir / "metadata.txt");
        metadata << "timestamp_utc=" << timestampUtc() << '\n'
                 << "gpu=" << prop.name << '\n'
                 << "compute_capability=" << prop.major << '.' << prop.minor << '\n'
                 << "cuda_driver_api=" << driverVersion << '\n'
                 << "cuda_runtime_api=" << runtimeVersion << '\n'
                 << "vram_total_bytes=" << totalBytes << '\n'
                 << "vram_free_before_bytes=" << freeBytes << '\n'
                 << "expert_bytes=" << kExpertBytes << '\n'
                 << "model_layers=" << kLayers << '\n'
                 << "experts_per_layer=" << kExpertsPerLayer << '\n'
                 << "routed_topk=" << kTopK << '\n'
                 << "compute_model=three_fp16_strided_batched_gemms_no_activation_or_int4_dequant\n";

        std::ofstream copyCsv(options.outputDir / "copy_results.csv");
        copyCsv << "pool_gb,host_memory,layout,batch_experts,bytes,iterations,"
                   "wall_mean_ms,wall_p50_ms,wall_p95_ms,event_mean_ms,effective_gbps\n";
        copyCsv << std::fixed << std::setprecision(6);

        std::cout << "GPU: " << prop.name << "  CC " << prop.major << '.' << prop.minor
                  << "  free " << std::setprecision(2) << (freeBytes / 1e9)
                  << "/" << (totalBytes / 1e9) << " GB\n";
        std::cout << "Expert record: " << kExpertBytes << " bytes; max batch host allocation: "
                  << (maxHostBytes / 1e6) << " MB\n";

        bool pipelineDone = false;
        std::ofstream overlapCsv(options.outputDir / "overlap_results.csv");
        overlapCsv << "pool_gb,batch_experts,bytes,pipeline_cycles,pipeline_rounds,"
                      "copy_only_ms,compute_only_ms,serial_ms,double_buffer_serial_ms,"
                      "overlap_mean_ms,overlap_p50_ms,overlap_p95_ms,exposed_copy_ms,"
                      "hidden_copy_percent,equivalent_gemm_flops\n";
        overlapCsv << std::fixed << std::setprecision(6);

        for (double requestedGB : options.poolGB) {
            const std::size_t requestedBytes = static_cast<std::size_t>(requestedGB * 1e9);
            CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
            if (requestedBytes + 512'000'000ull > freeBytes) {
                std::cout << "Skipping " << requestedGB << " GB pool: only "
                          << (freeBytes / 1e9) << " GB currently free\n";
                continue;
            }
            std::uint8_t* devicePool = nullptr;
            CUDA_CHECK(cudaMalloc(&devicePool, requestedBytes));
            CUDA_CHECK(cudaMemsetAsync(devicePool, 0, std::min<std::size_t>(requestedBytes, 64ull << 20), copyStream));
            CUDA_CHECK(cudaStreamSynchronize(copyStream));
            std::cout << "\nPool " << requestedGB << " GB\n";

            const int validationBatch = std::min(maxBatch, 2);
            validateCopyPath(Layout::ContiguousRecord, devicePool, pinned, validationBatch, copyStream);
            validateCopyPath(Layout::NinePiecesPerExpert, devicePool, pinned, validationBatch, copyStream);

            for (const char* hostName : {"pageable", "pinned"}) {
                const std::uint8_t* host = std::strcmp(hostName, "pageable") == 0 ? pageable.data() : pinned;
                for (Layout layout : {Layout::ContiguousRecord, Layout::NinePiecesPerExpert}) {
                    for (int batch : options.batches) {
                        const CopyStats stats = benchmarkCopy(
                            layout, devicePool, requestedBytes, host, batch, options, copyStream);
                        const std::size_t bytes = static_cast<std::size_t>(batch) * kExpertBytes;
                        copyCsv << requestedGB << ',' << hostName << ',' << layoutName(layout) << ','
                                << batch << ',' << bytes << ',' << stats.iterations << ','
                                << stats.wallMeanMs << ',' << stats.wallP50Ms << ','
                                << stats.wallP95Ms << ',' << stats.eventMeanMs << ','
                                << stats.effectiveGBs << '\n';
                        copyCsv.flush();
                        std::cout << "  " << std::setw(8) << hostName << "  "
                                  << std::setw(22) << layoutName(layout)
                                  << "  batch " << std::setw(2) << batch
                                  << "  p50 " << std::setw(7) << std::setprecision(3)
                                  << stats.wallP50Ms << " ms  p95 " << std::setw(7)
                                  << stats.wallP95Ms << " ms  " << std::setw(6)
                                  << std::setprecision(2) << stats.effectiveGBs << " GB/s\n";
                    }
                }
            }

            if (!pipelineDone && requestedGB >= 4.0) {
                EquivalentMoeCompute compute(maxBatch);
                std::cout << "  Running ideal known-next-batch overlap on this pool...\n";
                for (int batch : options.batches) {
                    const PipelineStats stats = benchmarkPipeline(
                        devicePool, requestedBytes, pinned, batch, options, compute);
                    const std::size_t bytes = static_cast<std::size_t>(batch) * kExpertBytes;
                    overlapCsv << requestedGB << ',' << batch << ',' << bytes << ','
                               << options.pipelineCycles << ',' << options.pipelineRounds << ','
                               << stats.copyOnlyMs << ',' << stats.computeOnlyMs << ','
                               << stats.serialMs << ',' << stats.doubleBufferedSerialMs << ','
                               << stats.overlapMeanMs << ',' << stats.overlapP50Ms << ','
                               << stats.overlapP95Ms << ',' << stats.exposedCopyMs << ','
                               << stats.hiddenCopyPercent << ','
                               << EquivalentMoeCompute::flopsPerExpert() * batch << '\n';
                    overlapCsv.flush();
                    std::cout << "    batch " << std::setw(2) << batch
                              << " copy " << std::setprecision(3) << stats.copyOnlyMs
                              << " ms, compute " << stats.computeOnlyMs
                              << " ms, serial " << stats.serialMs
                              << " ms, overlap " << stats.overlapMeanMs
                              << " ms, exposed " << stats.exposedCopyMs
                              << " ms, hidden " << std::setprecision(1)
                              << stats.hiddenCopyPercent << "%\n";
                }
                pipelineDone = true;
            }

            CUDA_CHECK(cudaFree(devicePool));
        }

        CUDA_CHECK(cudaStreamDestroy(copyStream));
        CUDA_CHECK(cudaFreeHost(pinned));
        CUDA_CHECK(cudaDeviceSynchronize());
        metadata << "copy_validation=pass\n";
        std::cout << "\nWrote results to " << options.outputDir.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
