#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> MAGIC = {'S', 'S', 'R', 'T', 'R', 'C', '0', '1'};
constexpr uint16_t VERSION = 1;
constexpr uint16_t HEADER_BYTES = 32;
constexpr int64_t ROUTED_TOP_K = 10;

enum class Phase : uint8_t {
    prefill = 0,
    decode = 1,
};

template<typename T>
void write_scalar(std::ostream & out, T value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

std::string json_escape(const std::string & input) {
    std::string out;
    out.reserve(input.size());
    for (unsigned char c : input) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open prompt file: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

int layer_from_name(const char * name) {
    const char * dash = std::strrchr(name, '-');
    return dash ? std::stoi(dash + 1) : -1;
}

bool starts_with(const char * value, const char * prefix) {
    return std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

std::vector<uint8_t> tensor_row_bytes(ggml_tensor * tensor, int64_t row, size_t size) {
    const size_t offset = row * tensor->nb[1];
    if (offset + size > ggml_nbytes(tensor)) {
        std::fprintf(stderr, "routing trace read out of bounds: %s row=%lld offset=%zu size=%zu nbytes=%zu ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
                tensor->name, (long long) row, offset, size, ggml_nbytes(tensor),
                (long long) tensor->ne[0], (long long) tensor->ne[1],
                (long long) tensor->ne[2], (long long) tensor->ne[3],
                tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
        throw std::runtime_error("routing trace tensor read out of bounds");
    }
    std::vector<uint8_t> bytes(size);
    ggml_backend_tensor_get(tensor, bytes.data(), offset, size);
    return bytes;
}

class TraceWriter {
public:
    explicit TraceWriter(const std::string & path, const std::string & hidden_path, int hidden_layer) :
            buffer_(4 * 1024 * 1024), hidden_layer_(hidden_layer) {
        out_.rdbuf()->pubsetbuf(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
        out_.open(path, std::ios::binary | std::ios::trunc);
        if (!out_) {
            throw std::runtime_error("failed to open trace: " + path);
        }
        write_header();
        if (!hidden_path.empty()) {
            hidden_.open(hidden_path, std::ios::binary | std::ios::trunc);
            if (!hidden_) {
                throw std::runtime_error("failed to open hidden-state trace: " + hidden_path);
            }
            write_hidden_header();
        }
    }

    ~TraceWriter() {
        try {
            finish();
        } catch (...) {
        }
    }

    void set_batch(Phase phase, uint32_t base_token, const std::vector<llama_token> & tokens) {
        phase_ = phase;
        base_token_ = base_token;
        batch_tokens_ = tokens;
        layer_offsets_.clear();
        pending_ids_.clear();
        pending_weights_.clear();
        completed_layers_.clear();
        hidden_tokens_seen_.clear();
    }

    bool callback(ggml_tensor * tensor, bool ask) {
        const bool ids = starts_with(tensor->name, "ffn_moe_topk-");
        const bool weights = starts_with(tensor->name, "ffn_moe_weights_norm-");
        const bool hidden = hidden_.is_open() && starts_with(tensor->name, "ffn_moe_inp-") &&
            layer_from_name(tensor->name) == hidden_layer_;
        if (ask) {
            return ids || weights || hidden;
        }
        if (ids) {
            collect_ids(tensor);
        } else if (weights) {
            collect_weights(tensor);
        } else if (hidden && phase_ == Phase::decode) {
            collect_hidden(tensor);
        }
        return true;
    }

    void finish() {
        if (finished_) {
            return;
        }
        finished_ = true;
        out_.seekp(0);
        write_header();
        out_.flush();
        out_.close();
        if (hidden_.is_open()) {
            hidden_.seekp(0);
            write_hidden_header();
            hidden_.flush();
            hidden_.close();
        }
    }

    uint32_t prompt_tokens() const { return prompt_tokens_; }
    uint32_t decode_tokens() const { return decode_tokens_; }
    uint16_t layers() const { return n_layers_; }
    uint16_t top_k() const { return top_k_; }
    uint64_t records() const { return records_; }

private:
    void write_hidden_header() {
        static constexpr std::array<char, 8> hidden_magic = {'S', 'S', 'H', 'I', 'D', '0', '0', '1'};
        hidden_.write(hidden_magic.data(), hidden_magic.size());
        write_scalar(hidden_, hidden_n_embd_);
        write_scalar(hidden_, hidden_rows_);
    }

    void write_header() {
        out_.write(MAGIC.data(), MAGIC.size());
        write_scalar(out_, VERSION);
        write_scalar(out_, HEADER_BYTES);
        const uint16_t record_bytes = static_cast<uint16_t>(12 + 6 * top_k_);
        write_scalar(out_, record_bytes);
        write_scalar(out_, n_layers_);
        write_scalar(out_, top_k_);
        write_scalar(out_, uint16_t{0});
        write_scalar(out_, prompt_tokens_);
        write_scalar(out_, decode_tokens_);
        write_scalar(out_, uint32_t{0});
    }

    void collect_ids(ggml_tensor * tensor) {
        if (tensor->type != GGML_TYPE_I32) {
            throw std::runtime_error("ffn_moe_topk has unexpected type");
        }
        const int layer = layer_from_name(tensor->name);
        const int64_t k = std::min<int64_t>(ROUTED_TOP_K, tensor->ne[0]);
        const int64_t n_tokens = tensor->ne[1];
        std::vector<int32_t> values(k * n_tokens);
        for (int64_t token = 0; token < n_tokens; ++token) {
            auto bytes = tensor_row_bytes(tensor, token, k * sizeof(int32_t));
            std::memcpy(values.data() + token * k, bytes.data(), bytes.size());
        }
        pending_ids_[layer] = std::move(values);
        top_k_ = static_cast<uint16_t>(k);
        n_layers_ = std::max<uint16_t>(n_layers_, static_cast<uint16_t>(layer + 1));
        emit_layer(layer);
    }

    void collect_weights(ggml_tensor * tensor) {
        if (tensor->type != GGML_TYPE_F32) {
            throw std::runtime_error("ffn_moe_weights_norm has unexpected type");
        }
        const int layer = layer_from_name(tensor->name);
        const int64_t n_tokens = tensor->ne[1];
        const int64_t k = tensor->ne[0];
        std::vector<float> weights(ggml_nelements(tensor));
        for (int64_t token = 0; token < n_tokens; ++token) {
            auto bytes = tensor_row_bytes(tensor, token, k * sizeof(float));
            std::memcpy(weights.data() + token * k, bytes.data(), bytes.size());
        }
        pending_weights_[layer] = std::move(weights);
        emit_layer(layer);
    }

    void emit_layer(int layer) {
        if (completed_layers_.count(layer)) {
            return;
        }
        auto ids = pending_ids_.find(layer);
        auto weights = pending_weights_.find(layer);
        if (ids == pending_ids_.end() || weights == pending_weights_.end()) {
            return;
        }
        if (top_k_ == 0 || ids->second.size() != weights->second.size() || ids->second.size() % top_k_ != 0) {
            throw std::runtime_error("routing tensor shape mismatch");
        }
        const int64_t k = top_k_;
        const int64_t n_tokens = ids->second.size() / k;

        uint32_t & layer_offset = layer_offsets_[layer];
        for (int64_t token = 0; token < n_tokens; ++token) {
            const uint32_t local = layer_offset + static_cast<uint32_t>(token);
            const uint32_t token_index = base_token_ + local;
            const int32_t token_id = local < batch_tokens_.size() ? batch_tokens_[local] : LLAMA_TOKEN_NULL;
            write_scalar(out_, token_index);
            write_scalar(out_, token_id);
            write_scalar(out_, static_cast<uint16_t>(layer));
            write_scalar(out_, static_cast<uint8_t>(phase_));
            write_scalar(out_, static_cast<uint8_t>(top_k_));
            const int64_t offset = token * k;
            for (int64_t expert = 0; expert < k; ++expert) {
                write_scalar(out_, static_cast<int16_t>(ids->second[offset + expert]));
            }
            for (int64_t expert = 0; expert < k; ++expert) {
                write_scalar(out_, weights->second[offset + expert]);
            }
            ++records_;
        }
        layer_offset += static_cast<uint32_t>(n_tokens);
        if (phase_ == Phase::prefill) {
            prompt_tokens_ = std::max(prompt_tokens_, layer_offset);
        } else {
            const uint32_t count = base_token_ >= prompt_tokens_ ? base_token_ - prompt_tokens_ + layer_offset : 0;
            decode_tokens_ = std::max(decode_tokens_, count);
        }
        pending_ids_.erase(ids);
        pending_weights_.erase(weights);
        completed_layers_.insert(layer);
    }

    void collect_hidden(ggml_tensor * tensor) {
        if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0) {
            throw std::runtime_error("hidden-state tensor has an invalid shape");
        }
        const uint32_t n_embd = static_cast<uint32_t>(tensor->ne[0]);
        if (hidden_n_embd_ != 0 && hidden_n_embd_ != n_embd) {
            throw std::runtime_error("hidden-state width changed during tracing");
        }
        hidden_n_embd_ = n_embd;
        const int64_t n_tokens = tensor->ne[1];
        const size_t source_element_bytes = ggml_type_size(tensor->type);
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
            throw std::runtime_error("ffn_moe_inp has an unsupported type");
        }
        std::vector<float> values(n_embd);
        for (int64_t token = 0; token < n_tokens; ++token) {
            const uint32_t token_index = base_token_ + static_cast<uint32_t>(token);
            if (!hidden_tokens_seen_.insert(token_index).second) {
                continue;
            }
            auto bytes = tensor_row_bytes(tensor, token, n_embd * source_element_bytes);
            if (tensor->type == GGML_TYPE_F32) {
                std::memcpy(values.data(), bytes.data(), n_embd * sizeof(float));
            } else if (tensor->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(bytes.data()), values.data(), n_embd);
            } else {
                ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(bytes.data()), values.data(), n_embd);
            }
            write_scalar(hidden_, token_index);
            hidden_.write(reinterpret_cast<const char *>(values.data()), n_embd * sizeof(float));
            ++hidden_rows_;
        }
    }

    std::ofstream out_;
    std::ofstream hidden_;
    std::vector<char> buffer_;
    bool finished_ = false;
    Phase phase_ = Phase::prefill;
    uint32_t base_token_ = 0;
    std::vector<llama_token> batch_tokens_;
    std::map<int, uint32_t> layer_offsets_;
    std::map<int, std::vector<int32_t>> pending_ids_;
    std::map<int, std::vector<float>> pending_weights_;
    std::set<int> completed_layers_;
    std::set<uint32_t> hidden_tokens_seen_;
    uint16_t n_layers_ = 0;
    uint16_t top_k_ = 0;
    uint32_t prompt_tokens_ = 0;
    uint32_t decode_tokens_ = 0;
    uint64_t records_ = 0;
    int hidden_layer_ = 0;
    uint32_t hidden_n_embd_ = 0;
    uint32_t hidden_rows_ = 0;
};

bool trace_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    try {
        return static_cast<TraceWriter *>(user_data)->callback(tensor, ask);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "routing callback error for %s: %s\n", tensor->name, error.what());
        return false;
    }
}

struct Options {
    std::string model;
    std::string trace;
    std::string meta;
    std::string prompt_file;
    std::string workload = "unspecified";
    int n_predict = 128;
    int n_gpu_layers = 8;
    int n_ctx = 4096;
    int n_threads = 24;
    int prompt_tokens = 0;
    std::string hidden;
    int hidden_layer = 0;
    bool cpu_moe = false;
};

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s -m MODEL --trace TRACE --meta META --prompt-file FILE [--prompt-tokens TOKENS] [--cpu-moe] [--hidden FILE] [--hidden-layer LAYER] [--workload NAME] [-n TOKENS] [-ngl LAYERS] [-c CTX] [-t THREADS]\n", argv0);
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        auto value = [&](const char * option) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + option);
            }
            return argv[++i];
        };
        if (std::strcmp(argv[i], "-m") == 0) options.model = value("-m");
        else if (std::strcmp(argv[i], "--trace") == 0) options.trace = value("--trace");
        else if (std::strcmp(argv[i], "--meta") == 0) options.meta = value("--meta");
        else if (std::strcmp(argv[i], "--prompt-file") == 0) options.prompt_file = value("--prompt-file");
        else if (std::strcmp(argv[i], "--hidden") == 0) options.hidden = value("--hidden");
        else if (std::strcmp(argv[i], "--hidden-layer") == 0) options.hidden_layer = std::stoi(value("--hidden-layer"));
        else if (std::strcmp(argv[i], "--workload") == 0) options.workload = value("--workload");
        else if (std::strcmp(argv[i], "-n") == 0) options.n_predict = std::stoi(value("-n"));
        else if (std::strcmp(argv[i], "-ngl") == 0) options.n_gpu_layers = std::stoi(value("-ngl"));
        else if (std::strcmp(argv[i], "-c") == 0) options.n_ctx = std::stoi(value("-c"));
        else if (std::strcmp(argv[i], "-t") == 0) options.n_threads = std::stoi(value("-t"));
        else if (std::strcmp(argv[i], "--prompt-tokens") == 0) options.prompt_tokens = std::stoi(value("--prompt-tokens"));
        else if (std::strcmp(argv[i], "--cpu-moe") == 0) options.cpu_moe = true;
        else throw std::runtime_error(std::string("unknown option: ") + argv[i]);
    }
    if (options.model.empty() || options.trace.empty() || options.meta.empty() || options.prompt_file.empty()) {
        throw std::runtime_error("required option is missing");
    }
    return options;
}

std::string apply_chat_template(const llama_model * model, const std::string & prompt) {
    const llama_chat_message message = {"user", prompt.c_str()};
    const char * tmpl = llama_model_chat_template(model, nullptr);
    int32_t size = llama_chat_apply_template(tmpl, &message, 1, true, nullptr, 0);
    if (size < 0) {
        throw std::runtime_error("chat template sizing failed");
    }
    std::vector<char> text(size + 1);
    size = llama_chat_apply_template(tmpl, &message, 1, true, text.data(), static_cast<int32_t>(text.size()));
    if (size < 0) {
        throw std::runtime_error("chat template application failed");
    }
    return std::string(text.data(), size);
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int32_t count = llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    if (count >= 0) {
        throw std::runtime_error("token count probe returned an unexpected value");
    }
    std::vector<llama_token> tokens(-count);
    count = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (count < 0) {
        throw std::runtime_error("tokenization failed");
    }
    tokens.resize(count);
    return tokens;
}

void write_meta(const Options & options, const std::string & prompt, int prompt_tokens, int decode_tokens, const TraceWriter & trace) {
    std::ofstream out(options.meta, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open metadata file");
    }
    out << "{\n"
        << "  \"format\": \"slotstream-routing-trace-v1\",\n"
        << "  \"workload\": \"" << json_escape(options.workload) << "\",\n"
        << "  \"model_path\": \"" << json_escape(options.model) << "\",\n"
        << "  \"runtime\": \"llama.cpp b10666 / 4e97ac86ebe2c4cb8212d98d2641ad6768810896\",\n"
        << "  \"prompt\": \"" << json_escape(prompt) << "\",\n"
        << "  \"prompt_tokens\": " << prompt_tokens << ",\n"
        << "  \"decode_tokens\": " << decode_tokens << ",\n"
        << "  \"seed\": 0,\n"
        << "  \"sampling\": {\"strategy\": \"greedy\"},\n"
        << "  \"routing\": {\"experts\": 512, \"top_k\": " << trace.top_k() << ", \"normalization\": \"top-k softmax\", \"shared_expert\": true},\n"
        << "  \"n_layers\": " << trace.layers() << ",\n"
        << "  \"records\": " << trace.records() << ",\n"
        << "  \"hidden_state_trace\": " << (options.hidden.empty() ? "null" : ("\"" + json_escape(options.hidden) + "\"")) << ",\n"
        << "  \"hidden_state_layer\": " << options.hidden_layer << ",\n"
        << "  \"n_gpu_layers\": " << options.n_gpu_layers << ",\n"
        << "  \"cpu_moe\": " << (options.cpu_moe ? "true" : "false") << ",\n"
        << "  \"threads\": " << options.n_threads << "\n"
        << "}\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::string prompt = read_file(options.prompt_file);

        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options.n_gpu_layers;
        static const char * cpu_moe_pattern = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
        std::vector<llama_model_tensor_buft_override> tensor_overrides;
        if (options.cpu_moe) {
            tensor_overrides.push_back({cpu_moe_pattern, ggml_backend_cpu_buffer_type()});
            tensor_overrides.push_back({nullptr, nullptr});
            model_params.tensor_buft_overrides = tensor_overrides.data();
        }
        llama_model * model = llama_model_load_from_file(options.model.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("model load failed");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const std::string formatted = apply_chat_template(model, prompt);
        std::vector<llama_token> prompt_tokens = tokenize(vocab, formatted);
        if (options.prompt_tokens > 0) {
            if (prompt_tokens.size() < 2) throw std::runtime_error("prompt seed is too short");
            const std::vector<llama_token> seed = prompt_tokens;
            prompt_tokens.resize(options.prompt_tokens);
            for (int i = static_cast<int>(seed.size()); i < options.prompt_tokens; ++i) {
                prompt_tokens[i] = seed[1 + (i - static_cast<int>(seed.size())) % (seed.size() - 1)];
            }
        }
        if (static_cast<int>(prompt_tokens.size()) + options.n_predict > options.n_ctx) {
            throw std::runtime_error("prompt plus decode length exceeds context size");
        }

        TraceWriter trace(options.trace, options.hidden, options.hidden_layer);
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = options.n_ctx;
        context_params.n_batch = std::min(options.n_ctx, 2048);
        context_params.n_ubatch = (std::min)(context_params.n_batch, 1024u);
        context_params.no_perf = false;
        context_params.cb_eval = trace_callback;
        context_params.cb_eval_user_data = &trace;
        llama_context * context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            throw std::runtime_error("context creation failed");
        }
        llama_set_n_threads(context, options.n_threads, options.n_threads);

        llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
        sampler_params.no_perf = false;
        llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

        trace.set_batch(Phase::prefill, 0, prompt_tokens);
        // n_tokens_all must not exceed n_batch, so submit the prompt in
        // n_batch-sized chunks. layer_offsets_ accumulates across calls, so
        // token positions in the trace stay contiguous for the whole prompt.
        for (size_t offset = 0; offset < prompt_tokens.size(); ) {
            const int32_t n_chunk = static_cast<int32_t>(
                std::min<size_t>(context_params.n_batch, prompt_tokens.size() - offset));
            llama_batch batch = llama_batch_get_one(prompt_tokens.data() + offset, n_chunk);
            if (llama_decode(context, batch) != 0) {
                throw std::runtime_error("prefill evaluation failed");
            }
            offset += static_cast<size_t>(n_chunk);
        }

        llama_token token = llama_sampler_sample(sampler, context, -1);
        int decoded = 0;
        for (; decoded < options.n_predict && !llama_vocab_is_eog(vocab, token); ++decoded) {
            char piece[256];
            const int size = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
            if (size > 0) {
                std::cout.write(piece, size);
                std::cout.flush();
            }
            std::vector<llama_token> input = {token};
            trace.set_batch(Phase::decode, static_cast<uint32_t>(prompt_tokens.size() + decoded), input);
            llama_batch batch = llama_batch_get_one(input.data(), 1);
            if (llama_decode(context, batch) != 0) {
                throw std::runtime_error("decode evaluation failed");
            }
            token = llama_sampler_sample(sampler, context, -1);
        }
        std::cout << '\n';

        trace.finish();
        write_meta(options, prompt, static_cast<int>(prompt_tokens.size()), decoded, trace);
        std::fprintf(stderr, "trace: prompt=%zu decode=%d layers=%u top_k=%u records=%llu\n", prompt_tokens.size(), decoded, trace.layers(), trace.top_k(), static_cast<unsigned long long>(trace.records()));
        llama_perf_context_print(context);

        llama_sampler_free(sampler);
        llama_free(context);
        llama_model_free(model);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "slotstream-routing-trace: %s\n", error.what());
        usage(argv[0]);
        return 1;
    }
}
