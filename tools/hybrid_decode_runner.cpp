#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;

struct Options {
    std::string model;
    std::string state_file;
    std::string output_json;
    std::string logits_file;
    std::string final_logits_file;
    std::string routing_json;
    std::string input_tokens_file;
    bool routing_ids_only = false;
    std::string hybrid_cache_plan;
    int promote_threshold = -1;
    int verify_tokens = 32;
    int warmup_tokens = 100;
    int measured_tokens = 500;
    int n_ctx = 17408;
    int n_threads = 24;
    int n_gpu_layers = -2;
    int n_batch = 2048;
    int n_ubatch = 16;
    std::string cache_type_k = "f16";
    std::string cache_type_v = "f16";
};

double elapsed_ms(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

ggml_type cache_type_from_name(const std::string & name) {
    if (name == "f16") return GGML_TYPE_F16;
    if (name == "q8_0") return GGML_TYPE_Q8_0;
    if (name == "q4_0") return GGML_TYPE_Q4_0;
    throw std::runtime_error("invalid KV cache type: " + name);
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        auto value = [&](const char * option) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + option);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "-m") == 0) options.model = value("-m");
        else if (std::strcmp(argv[i], "--state-file") == 0) options.state_file = value("--state-file");
        else if (std::strcmp(argv[i], "--output-json") == 0) options.output_json = value("--output-json");
        else if (std::strcmp(argv[i], "--logits-file") == 0) options.logits_file = value("--logits-file");
        else if (std::strcmp(argv[i], "--final-logits-file") == 0) options.final_logits_file = value("--final-logits-file");
        else if (std::strcmp(argv[i], "--routing-json") == 0) options.routing_json = value("--routing-json");
        else if (std::strcmp(argv[i], "--input-tokens-file") == 0) options.input_tokens_file = value("--input-tokens-file");
        else if (std::strcmp(argv[i], "--routing-ids-only") == 0) options.routing_ids_only = true;
        else if (std::strcmp(argv[i], "--hybrid-cache-plan") == 0) options.hybrid_cache_plan = value("--hybrid-cache-plan");
        else if (std::strcmp(argv[i], "--promote-threshold") == 0) options.promote_threshold = std::stoi(value("--promote-threshold"));
        else if (std::strcmp(argv[i], "--verify-tokens") == 0) options.verify_tokens = std::stoi(value("--verify-tokens"));
        else if (std::strcmp(argv[i], "--warmup-tokens") == 0) options.warmup_tokens = std::stoi(value("--warmup-tokens"));
        else if (std::strcmp(argv[i], "--measured-tokens") == 0) options.measured_tokens = std::stoi(value("--measured-tokens"));
        else if (std::strcmp(argv[i], "-c") == 0) options.n_ctx = std::stoi(value("-c"));
        else if (std::strcmp(argv[i], "-t") == 0) options.n_threads = std::stoi(value("-t"));
        else if (std::strcmp(argv[i], "-ngl") == 0) options.n_gpu_layers = std::stoi(value("-ngl"));
        else if (std::strcmp(argv[i], "--n-batch") == 0) options.n_batch = std::stoi(value("--n-batch"));
        else if (std::strcmp(argv[i], "--n-ubatch") == 0) options.n_ubatch = std::stoi(value("--n-ubatch"));
        else if (std::strcmp(argv[i], "--cache-type-k") == 0) options.cache_type_k = value("--cache-type-k");
        else if (std::strcmp(argv[i], "--cache-type-v") == 0) options.cache_type_v = value("--cache-type-v");
        else throw std::runtime_error(std::string("unknown option: ") + argv[i]);
    }
    if (options.model.empty() || options.state_file.empty() || options.output_json.empty()) {
        throw std::runtime_error("required option is missing");
    }
    if (options.verify_tokens < 1 || options.warmup_tokens < 0 || options.measured_tokens < 1 ||
            options.n_batch < 1 || options.n_ubatch < 1 || options.n_ubatch > options.n_batch ||
            options.promote_threshold < -1 || options.promote_threshold > 10) {
        throw std::runtime_error("invalid measurement options");
    }
    cache_type_from_name(options.cache_type_k);
    cache_type_from_name(options.cache_type_v);
    return options;
}

llama_token sample_greedy(llama_context * context) {
    llama_sampler * sampler = llama_sampler_init_greedy();
    const llama_token token = llama_sampler_sample(sampler, context, -1);
    llama_sampler_free(sampler);
    return token;
}

void replay_last_token(llama_context * context, llama_token token, int32_t position) {
    llama_batch batch = llama_batch_get_one(&token, 1);
    batch.pos = &position;
    if (llama_decode(context, batch) != 0) {
        throw std::runtime_error("last-token replay failed");
    }
}

void decode_one(llama_context * context, llama_token token) {
    if (llama_decode(context, llama_batch_get_one(&token, 1)) != 0) {
        throw std::runtime_error("decode failed");
    }
}

std::string json_string(const std::string & value) {
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('"');
    for (char ch : value) {
        if (ch == '\\' || ch == '"') result.push_back('\\');
        result.push_back(ch);
    }
    result.push_back('"');
    return result;
}

void write_tokens(std::ostream & output, const std::vector<llama_token> & tokens) {
    output << '[';
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) output << ',';
        output << tokens[i];
    }
    output << ']';
}

bool starts_with(const char * value, const char * prefix) {
    return std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

int layer_from_name(const char * name) {
    const char * dash = std::strrchr(name, '-');
    return dash == nullptr ? -1 : std::stoi(dash + 1);
}

struct RoutingEventI32 {
    int layer;
    std::vector<int32_t> values;
};

struct RoutingEventF32 {
    int layer;
    std::vector<float> values;
};

class RoutingCapture {
public:
    RoutingCapture(const std::string & plan_path, bool capture_weights) :
            resident_(48, std::vector<uint8_t>(512, 0)), capture_weights_(capture_weights) {
        if (plan_path.empty()) return;
        std::ifstream input(plan_path);
        if (!input) throw std::runtime_error("routing capture cache plan open failed");
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream row(line);
            int layer = -1;
            int expert = -1;
            if (!(row >> layer >> expert)) continue;
            if (layer >= 0 && layer < 48 && expert >= 0 && expert < 512) {
                resident_[layer][expert] = 1;
                has_plan_ = true;
            }
        }
    }

    void begin_decode() {
        ids_seen_.assign(48, 0);
        weights_seen_.assign(48, 0);
    }

    bool callback(ggml_tensor * tensor, bool ask) {
        const bool ids = starts_with(tensor->name, "ffn_moe_topk-");
        const bool weights = capture_weights_ && starts_with(tensor->name, "ffn_moe_weights_norm-");
        if (ask) return ids || weights;
        const int layer = layer_from_name(tensor->name);
        if (ids) {
            if (layer < 0 || layer >= 48 || ids_seen_[layer]) return true;
            ids_seen_[layer] = 1;
            if (tensor->type != GGML_TYPE_I32) throw std::runtime_error("routing ids type mismatch");
            const int64_t k = tensor->ne[0];
            for (int64_t row = 0; row < tensor->ne[1]; ++row) {
                RoutingEventI32 event{layer, std::vector<int32_t>(k)};
                ggml_backend_tensor_get(tensor, event.values.data(), row * tensor->nb[1], k * sizeof(int32_t));
                int misses = 0;
                for (int32_t expert : event.values) {
                    ++total_selected_;
                    if (has_plan_ && layer >= 0 && layer < 48 && expert >= 0 && expert < 512 && resident_[layer][expert]) {
                        ++cache_hits_;
                    } else if (has_plan_) {
                        ++misses;
                    }
                }
                if (has_plan_) {
                    miss_histogram_[(std::min)(10, misses)]++;
                    if (misses != 0) ++fallback_layers_;
                }
                ids_.push_back(std::move(event));
            }
        } else if (weights) {
            if (layer < 0 || layer >= 48 || weights_seen_[layer]) return true;
            weights_seen_[layer] = 1;
            if (tensor->type != GGML_TYPE_F32) throw std::runtime_error("routing weights type mismatch");
            const int64_t k = tensor->ne[0];
            for (int64_t row = 0; row < tensor->ne[1]; ++row) {
                RoutingEventF32 event{layer, std::vector<float>(k)};
                ggml_backend_tensor_get(tensor, event.values.data(), row * tensor->nb[1], k * sizeof(float));
                weights_.push_back(std::move(event));
            }
        }
        return true;
    }

    void write(const std::string & path) const {
        if (path.empty()) return;
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("routing JSON open failed");
        output << std::setprecision(9);
        output << "{\n  \"evidence\": \"Measured instrumented\",\n";
        output << "  \"id_events\": " << ids_.size() << ",\n";
        output << "  \"weight_events\": " << weights_.size() << ",\n";
        output << "  \"selected_experts\": " << total_selected_ << ",\n";
        output << "  \"cache_hits\": " << cache_hits_ << ",\n";
        output << "  \"cache_misses\": " << (has_plan_ ? total_selected_ - cache_hits_ : 0) << ",\n";
        output << "  \"hit_rate\": " << (has_plan_ && total_selected_ ? double(cache_hits_) / total_selected_ : 0.0) << ",\n";
        output << "  \"cpu_fallback_layers\": " << fallback_layers_ << ",\n";
        output << "  \"cpu_fallback_layers_per_token\": " << (ids_.empty() ? 0.0 : double(fallback_layers_) / (ids_.size() / 48.0)) << ",\n";
        output << "  \"miss_count_histogram\": [";
        for (size_t i = 0; i < miss_histogram_.size(); ++i) {
            if (i != 0) output << ',';
            output << miss_histogram_[i];
        }
        output << "],\n  \"id_layers\": [";
        for (size_t i = 0; i < ids_.size(); ++i) {
            if (i != 0) output << ',';
            output << ids_[i].layer;
        }
        output << "],\n  \"selected_ids\": [";
        bool first = true;
        for (const auto & event : ids_) for (int32_t value : event.values) {
            if (!first) output << ',';
            first = false;
            output << value;
        }
        output << "],\n  \"weight_layers\": [";
        for (size_t i = 0; i < weights_.size(); ++i) {
            if (i != 0) output << ',';
            output << weights_[i].layer;
        }
        output << "],\n  \"routing_weights\": [";
        first = true;
        for (const auto & event : weights_) for (float value : event.values) {
            if (!first) output << ',';
            first = false;
            output << value;
        }
        output << "]\n}\n";
    }

private:
    std::vector<std::vector<uint8_t>> resident_;
    bool has_plan_ = false;
    bool capture_weights_ = true;
    uint64_t total_selected_ = 0;
    uint64_t cache_hits_ = 0;
    uint64_t fallback_layers_ = 0;
    std::vector<uint64_t> miss_histogram_ = std::vector<uint64_t>(11, 0);
    std::vector<RoutingEventI32> ids_;
    std::vector<RoutingEventF32> weights_;
    std::vector<uint8_t> ids_seen_ = std::vector<uint8_t>(48, 0);
    std::vector<uint8_t> weights_seen_ = std::vector<uint8_t>(48, 0);
};

bool routing_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    try {
        return static_cast<RoutingCapture *>(user_data)->callback(tensor, ask);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "HYBRID_ROUTING_ERROR tensor=%s error=%s\n", tensor->name, error.what());
        return false;
    }
}
} // namespace

int main(int argc, char ** argv) {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    try {
        const Options options = parse_options(argc, argv);
        ggml_backend_load_all();
        RoutingCapture routing_capture(options.hybrid_cache_plan, !options.routing_ids_only);
        std::vector<llama_token> forced_tokens;
        if (!options.input_tokens_file.empty()) {
            std::ifstream input(options.input_tokens_file);
            int64_t value = 0;
            while (input >> value) forced_tokens.push_back(static_cast<llama_token>(value));
            if (!input.eof()) throw std::runtime_error("input token sequence parse failed");
        }

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options.n_gpu_layers;
        static const char * cpu_moe_pattern = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
        const llama_model_tensor_buft_override tensor_overrides[] = {
            {cpu_moe_pattern, ggml_backend_cpu_buffer_type()},
            {nullptr, nullptr},
        };
        model_params.tensor_buft_overrides = tensor_overrides;
        if (!options.hybrid_cache_plan.empty()) {
            model_params.hybrid_expert_cache_plan = options.hybrid_cache_plan.c_str();
            model_params.hybrid_expert_cache_promote_threshold = options.promote_threshold;
        }
        const auto model_begin = clock_type::now();
        model = llama_model_load_from_file(options.model.c_str(), model_params);
        const auto model_end = clock_type::now();
        if (model == nullptr) throw std::runtime_error("model load failed");

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = options.n_ctx;
        context_params.n_batch = (std::min)(static_cast<uint32_t>(options.n_ctx), static_cast<uint32_t>(options.n_batch));
        context_params.n_ubatch = (std::min)(context_params.n_batch, static_cast<uint32_t>(options.n_ubatch));
        context_params.type_k = cache_type_from_name(options.cache_type_k);
        context_params.type_v = cache_type_from_name(options.cache_type_v);
        context_params.no_perf = false;
        if (!options.routing_json.empty()) {
            context_params.cb_eval = routing_callback;
            context_params.cb_eval_user_data = &routing_capture;
        }
        const auto context_begin = clock_type::now();
        context = llama_init_from_model(model, context_params);
        const auto context_end = clock_type::now();
        if (context == nullptr) throw std::runtime_error("context creation failed");
        llama_set_n_threads(context, options.n_threads, options.n_threads);

        std::vector<llama_token> state_tokens(options.n_ctx);
        size_t state_token_count = 0;
        const auto load_begin = clock_type::now();
        if (!llama_state_load_file(context, options.state_file.c_str(), state_tokens.data(), state_tokens.size(), &state_token_count)) {
            throw std::runtime_error("state load failed");
        }
        const auto load_end = clock_type::now();
        state_tokens.resize(state_token_count);
        const int total_new_tokens = options.verify_tokens + options.warmup_tokens + options.measured_tokens;
        if (state_tokens.empty() || static_cast<int>(state_tokens.size()) + total_new_tokens >= options.n_ctx) {
            throw std::runtime_error("state plus measurement exceeds context");
        }
        if (!forced_tokens.empty() && static_cast<int>(forced_tokens.size()) < total_new_tokens) {
            throw std::runtime_error("input token sequence is too short");
        }

        const auto replay_begin = clock_type::now();
        routing_capture.begin_decode();
        replay_last_token(context, state_tokens.back(), static_cast<int32_t>(state_tokens.size() - 1));
        const auto replay_end = clock_type::now();

        const int vocabulary_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * logits = llama_get_logits(context);
        if (logits == nullptr) throw std::runtime_error("initial logits are missing");
        if (!options.logits_file.empty()) {
            std::ofstream logits_output(options.logits_file, std::ios::binary | std::ios::trunc);
            if (!logits_output) throw std::runtime_error("logits file open failed");
            logits_output.write(reinterpret_cast<const char *>(logits), vocabulary_size * sizeof(float));
        }

        llama_token token = sample_greedy(context);
        size_t forced_index = 0;
        std::vector<llama_token> continuation;
        continuation.reserve(total_new_tokens);
        for (int i = 0; i < options.verify_tokens; ++i) {
            if (!forced_tokens.empty()) token = forced_tokens[forced_index++];
            continuation.push_back(token);
            routing_capture.begin_decode();
            decode_one(context, token);
            token = sample_greedy(context);
        }

        const auto warmup_begin = clock_type::now();
        for (int i = 0; i < options.warmup_tokens; ++i) {
            if (!forced_tokens.empty()) token = forced_tokens[forced_index++];
            continuation.push_back(token);
            routing_capture.begin_decode();
            decode_one(context, token);
            token = sample_greedy(context);
        }
        const auto warmup_end = clock_type::now();

        if (options.promote_threshold >= 0) {
            llama_hybrid_cache_stats_reset(context);
        }

        std::vector<double> token_ms;
        token_ms.reserve(options.measured_tokens);
        const auto measured_begin = clock_type::now();
        for (int i = 0; i < options.measured_tokens; ++i) {
            if (!forced_tokens.empty()) token = forced_tokens[forced_index++];
            continuation.push_back(token);
            const auto token_begin = clock_type::now();
            routing_capture.begin_decode();
            decode_one(context, token);
            token = sample_greedy(context);
            const auto token_end = clock_type::now();
            token_ms.push_back(elapsed_ms(token_begin, token_end));
        }
        const auto measured_end = clock_type::now();

        if (!options.final_logits_file.empty()) {
            const float * final_logits = llama_get_logits(context);
            if (final_logits == nullptr) throw std::runtime_error("final logits are missing");
            std::ofstream final_logits_output(options.final_logits_file, std::ios::binary | std::ios::trunc);
            if (!final_logits_output) throw std::runtime_error("final logits file open failed");
            final_logits_output.write(reinterpret_cast<const char *>(final_logits), vocabulary_size * sizeof(float));
        }

        std::vector<double> ordered = token_ms;
        std::sort(ordered.begin(), ordered.end());
        const double measured_ms = elapsed_ms(measured_begin, measured_end);
        const double p50_ms = ordered[ordered.size() / 2];
        const size_t p95_index = (std::min)(ordered.size() - 1, static_cast<size_t>(ordered.size() * 0.95));
        const char * graph_disabled = std::getenv("GGML_CUDA_DISABLE_GRAPHS");
        const bool cuda_graphs = graph_disabled == nullptr || std::strcmp(graph_disabled, "1") != 0;
        const llama_hybrid_cache_stats cache_stats = llama_hybrid_cache_stats_get(context);

        std::ofstream output(options.output_json, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("output JSON open failed");
        output << std::setprecision(12);
        output << "{\n";
        output << "  \"evidence\": \"Measured\",\n";
        const char * mode = options.hybrid_cache_plan.empty() ? "native" :
                options.promote_threshold > 0 ? "hybrid-predictive-promotion" :
                options.promote_threshold == 0 ? "hybrid-instrumented-fixed-cache" : "hybrid-fixed-cache";
        output << "  \"mode\": \"" << mode << "\",\n";
        output << "  \"revision\": \"2d4f3154a2d93c3a4d6d4a415c404f1b397d8dcb\",\n";
        output << "  \"state_file\": " << json_string(options.state_file) << ",\n";
        output << "  \"cache_plan\": " << (options.hybrid_cache_plan.empty() ? "null" : json_string(options.hybrid_cache_plan)) << ",\n";
        output << "  \"promote_threshold\": " << options.promote_threshold << ",\n";
        output << "  \"forced_input_tokens\": " << (forced_tokens.empty() ? "false" : "true") << ",\n";
        output << "  \"cuda_graphs\": " << (cuda_graphs ? "true" : "false") << ",\n";
        output << "  \"n_ctx\": " << options.n_ctx << ",\n";
        output << "  \"n_batch\": " << context_params.n_batch << ",\n";
        output << "  \"n_ubatch\": " << context_params.n_ubatch << ",\n";
        output << "  \"cache_type_k\": " << json_string(options.cache_type_k) << ",\n";
        output << "  \"cache_type_v\": " << json_string(options.cache_type_v) << ",\n";
        output << "  \"state_tokens\": " << state_tokens.size() << ",\n";
        output << "  \"verify_tokens\": " << options.verify_tokens << ",\n";
        output << "  \"warmup_tokens\": " << options.warmup_tokens << ",\n";
        output << "  \"measured_tokens\": " << options.measured_tokens << ",\n";
        output << "  \"model_load_ms\": " << elapsed_ms(model_begin, model_end) << ",\n";
        output << "  \"context_init_ms\": " << elapsed_ms(context_begin, context_end) << ",\n";
        output << "  \"state_load_ms\": " << elapsed_ms(load_begin, load_end) << ",\n";
        output << "  \"last_token_replay_ms\": " << elapsed_ms(replay_begin, replay_end) << ",\n";
        output << "  \"warmup_ms\": " << elapsed_ms(warmup_begin, warmup_end) << ",\n";
        output << "  \"decode_ms\": " << measured_ms << ",\n";
        output << "  \"decode_ms_per_token\": " << measured_ms / options.measured_tokens << ",\n";
        output << "  \"decode_tok_s\": " << options.measured_tokens * 1000.0 / measured_ms << ",\n";
        output << "  \"decode_p50_ms\": " << p50_ms << ",\n";
        output << "  \"decode_p95_ms\": " << ordered[p95_index] << ",\n";
        if (options.promote_threshold >= 0) {
            const double stat_tokens = cache_stats.decode_tokens ? double(cache_stats.decode_tokens) : 1.0;
            output << "  \"cache_runtime_stats\": {\n";
            output << "    \"evidence\": \"Measured\",\n";
            output << "    \"decode_tokens\": " << cache_stats.decode_tokens << ",\n";
            output << "    \"layer_decisions\": " << cache_stats.layer_decisions << ",\n";
            output << "    \"expert_requests\": " << cache_stats.expert_requests << ",\n";
            output << "    \"cache_hits\": " << cache_stats.cache_hits << ",\n";
            output << "    \"cache_misses\": " << cache_stats.cache_misses << ",\n";
            output << "    \"hit_rate\": " << (cache_stats.expert_requests ? double(cache_stats.cache_hits) / cache_stats.expert_requests : 0.0) << ",\n";
            output << "    \"cpu_fallback_layers_per_token\": " << double(cache_stats.cpu_fallback_layers) / stat_tokens << ",\n";
            output << "    \"cpu_fallback_experts_per_token\": " << double(cache_stats.cache_misses) / stat_tokens << ",\n";
            output << "    \"h2d_experts_per_token\": " << double(cache_stats.promoted_experts) / stat_tokens << ",\n";
            output << "    \"h2d_weight_mb_per_token\": " << double(cache_stats.h2d_weight_bytes) / stat_tokens / 1000000.0 << ",\n";
            output << "    \"h2d_metadata_mb_per_token\": " << double(cache_stats.h2d_metadata_bytes) / stat_tokens / 1000000.0 << ",\n";
            output << "    \"promotion_ms_per_token\": " << double(cache_stats.promotion_time_us) / stat_tokens / 1000.0 << ",\n";
            output << "    \"miss_count_histogram\": [";
            for (int i = 0; i <= 10; ++i) {
                if (i != 0) output << ',';
                output << cache_stats.miss_count_histogram[i];
            }
            output << "]\n  },\n";
        }
        output << "  \"first_token\": " << continuation.front() << ",\n";
        output << "  \"verification_sequence\": ";
        write_tokens(output, std::vector<llama_token>(continuation.begin(), continuation.begin() + options.verify_tokens));
        output << ",\n  \"all_continuation_tokens\": ";
        write_tokens(output, continuation);
        output << "\n}\n";
        routing_capture.write(options.routing_json);

        std::fprintf(stderr,
            "HYBRID_DECODE mode=%s state_tokens=%zu verify=%d warmup=%d measured=%d tok_s=%.3f p50_ms=%.3f p95_ms=%.3f graph=%s\n",
            mode, state_tokens.size(),
            options.verify_tokens, options.warmup_tokens, options.measured_tokens,
            options.measured_tokens * 1000.0 / measured_ms, p50_ms, ordered[p95_index], cuda_graphs ? "on" : "off");

        llama_free(context);
        llama_model_free(model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "HYBRID_DECODE_ERROR %s\n", error.what());
        if (context != nullptr) llama_free(context);
        if (model != nullptr) llama_model_free(model);
        return 1;
    }
}
