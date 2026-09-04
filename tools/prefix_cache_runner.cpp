#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;

struct Options {
    std::string model;
    std::string prompt_file;
    std::string state_file;
    std::string output_json;
    std::string logits_file;
    std::string hybrid_cache_plan;
    int prompt_tokens = 16363;
    int measured_tokens = 100;
    int verify_tokens = 32;
    int n_ctx = 17408;
    int n_threads = 24;
    int n_gpu_layers = -2;
    int n_batch = 2048;
    int n_ubatch = 512;
    std::string cache_type_k = "f16";
    std::string cache_type_v = "f16";
    bool cpu_moe = true;
    bool load_only = false;
    bool cold_only = false;
};

ggml_type cache_type_from_name(const std::string & name) {
    if (name == "f16") return GGML_TYPE_F16;
    if (name == "q8_0") return GGML_TYPE_Q8_0;
    if (name == "q4_0") return GGML_TYPE_Q4_0;
    throw std::runtime_error("invalid KV cache type: " + name);
}

double elapsed_ms(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        auto value = [&](const char * option) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + option);
            return argv[++i];
        };
        if (std::strcmp(argv[i], "-m") == 0) options.model = value("-m");
        else if (std::strcmp(argv[i], "--prompt-file") == 0) options.prompt_file = value("--prompt-file");
        else if (std::strcmp(argv[i], "--state-file") == 0) options.state_file = value("--state-file");
        else if (std::strcmp(argv[i], "--output-json") == 0) options.output_json = value("--output-json");
        else if (std::strcmp(argv[i], "--logits-file") == 0) options.logits_file = value("--logits-file");
        else if (std::strcmp(argv[i], "--hybrid-cache-plan") == 0) options.hybrid_cache_plan = value("--hybrid-cache-plan");
        else if (std::strcmp(argv[i], "--prompt-tokens") == 0) options.prompt_tokens = std::stoi(value("--prompt-tokens"));
        else if (std::strcmp(argv[i], "--measured-tokens") == 0) options.measured_tokens = std::stoi(value("--measured-tokens"));
        else if (std::strcmp(argv[i], "--verify-tokens") == 0) options.verify_tokens = std::stoi(value("--verify-tokens"));
        else if (std::strcmp(argv[i], "-c") == 0) options.n_ctx = std::stoi(value("-c"));
        else if (std::strcmp(argv[i], "-t") == 0) options.n_threads = std::stoi(value("-t"));
        else if (std::strcmp(argv[i], "-ngl") == 0) options.n_gpu_layers = std::stoi(value("-ngl"));
        else if (std::strcmp(argv[i], "--n-batch") == 0) options.n_batch = std::stoi(value("--n-batch"));
        else if (std::strcmp(argv[i], "--n-ubatch") == 0) options.n_ubatch = std::stoi(value("--n-ubatch"));
        else if (std::strcmp(argv[i], "--cache-type-k") == 0) options.cache_type_k = value("--cache-type-k");
        else if (std::strcmp(argv[i], "--cache-type-v") == 0) options.cache_type_v = value("--cache-type-v");
        else if (std::strcmp(argv[i], "--no-cpu-moe") == 0) options.cpu_moe = false;
        else if (std::strcmp(argv[i], "--load-only") == 0) options.load_only = true;
        else if (std::strcmp(argv[i], "--cold-only") == 0) options.cold_only = true;
        else throw std::runtime_error(std::string("unknown option: ") + argv[i]);
    }
    if (options.model.empty() || options.prompt_file.empty() || options.state_file.empty() || options.output_json.empty()) {
        throw std::runtime_error("required option is missing");
    }
    const int verification_budget = options.load_only ? 0 : options.verify_tokens;
    if (options.load_only && options.cold_only) throw std::runtime_error("execution mode is ambiguous");
    if (options.n_batch < 1 || options.n_ubatch < 1 || options.n_ubatch > options.n_batch) {
        throw std::runtime_error("invalid batch size");
    }
    if (options.prompt_tokens < 2 || options.measured_tokens < 1 || options.verify_tokens < 1 ||
            options.prompt_tokens + options.measured_tokens + verification_budget >= options.n_ctx) {
        throw std::runtime_error("invalid token or context count");
    }
    cache_type_from_name(options.cache_type_k);
    cache_type_from_name(options.cache_type_v);
    return options;
}

std::string read_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("prompt file open failed");
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string apply_chat_template(const llama_model * model, const std::string & prompt) {
    const llama_chat_message message = {"user", prompt.c_str()};
    const char * tmpl = llama_model_chat_template(model, nullptr);
    int32_t size = llama_chat_apply_template(tmpl, &message, 1, true, nullptr, 0);
    if (size < 0) throw std::runtime_error("chat template sizing failed");
    std::vector<char> text(size + 1);
    size = llama_chat_apply_template(tmpl, &message, 1, true, text.data(), static_cast<int32_t>(text.size()));
    if (size < 0) throw std::runtime_error("chat template application failed");
    return std::string(text.data(), size);
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int32_t count = llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    if (count >= 0) throw std::runtime_error("token count probe failed");
    std::vector<llama_token> tokens(-count);
    count = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (count < 0) throw std::runtime_error("tokenization failed");
    tokens.resize(count);
    return tokens;
}

std::vector<llama_token> expand_prompt(std::vector<llama_token> seed, int count) {
    if (seed.size() < 2) throw std::runtime_error("prompt seed is too short");
    const size_t seed_size = seed.size();
    seed.resize(count);
    for (int i = static_cast<int>(seed_size); i < count; ++i) {
        seed[i] = seed[1 + (i - static_cast<int>(seed_size)) % (seed_size - 1)];
    }
    return seed;
}

llama_context * create_context(llama_model * model, const Options & options, double & init_ms) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = options.n_ctx;
    params.n_batch = (std::min)(static_cast<uint32_t>(options.n_ctx), static_cast<uint32_t>(options.n_batch));
    params.n_ubatch = (std::min)(params.n_batch, static_cast<uint32_t>(options.n_ubatch));
    params.type_k = cache_type_from_name(options.cache_type_k);
    params.type_v = cache_type_from_name(options.cache_type_v);
    params.no_perf = false;
    const auto begin = clock_type::now();
    llama_context * context = llama_init_from_model(model, params);
    const auto end = clock_type::now();
    init_ms = elapsed_ms(begin, end);
    if (context == nullptr) throw std::runtime_error("context creation failed");
    llama_set_n_threads(context, options.n_threads, options.n_threads);
    return context;
}

void decode_tokens(llama_context * context, llama_token * tokens, int count, int batch_size) {
    for (int offset = 0; offset < count; offset += batch_size) {
        const int current = (std::min)(batch_size, count - offset);
        if (llama_decode(context, llama_batch_get_one(tokens + offset, current)) != 0) {
            throw std::runtime_error("prompt decode failed");
        }
    }
}

void replay_last_token(llama_context * context, llama_token token, int32_t position) {
    llama_batch batch = llama_batch_get_one(&token, 1);
    batch.pos = &position;
    if (llama_decode(context, batch) != 0) throw std::runtime_error("last-token replay failed");
}

llama_token sample_greedy(llama_context * context) {
    llama_sampler * sampler = llama_sampler_init_greedy();
    const llama_token token = llama_sampler_sample(sampler, context, -1);
    llama_sampler_free(sampler);
    return token;
}

std::vector<float> copy_logits(llama_context * context, int vocabulary_size) {
    const float * logits = llama_get_logits(context);
    if (logits == nullptr) throw std::runtime_error("logits are missing");
    return std::vector<float>(logits, logits + vocabulary_size);
}

std::vector<llama_token> generate_tokens(llama_context * context, int count) {
    std::vector<llama_token> result;
    result.reserve(count);
    llama_token token = sample_greedy(context);
    for (int i = 0; i < count; ++i) {
        result.push_back(token);
        if (llama_decode(context, llama_batch_get_one(&token, 1)) != 0) {
            throw std::runtime_error("verification decode failed");
        }
        token = sample_greedy(context);
    }
    return result;
}

std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    if (tokens.empty()) return {};
    std::string text(tokens.size(), '\0');
    int32_t size = llama_detokenize(
        vocab, tokens.data(), static_cast<int32_t>(tokens.size()), text.data(), static_cast<int32_t>(text.size()),
        false, true);
    if (size < 0) {
        text.resize(-size);
        size = llama_detokenize(
            vocab, tokens.data(), static_cast<int32_t>(tokens.size()), text.data(), static_cast<int32_t>(text.size()),
            false, true);
    }
    if (size < 0) throw std::runtime_error("detokenization failed");
    text.resize(size);
    return text;
}

std::string json_string(const std::string & value) {
    static const char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('"');
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\': result += "\\\\"; break;
            case '"':  result += "\\\""; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (ch < 0x20) {
                    result += "\\u00";
                    result.push_back(hex[ch >> 4]);
                    result.push_back(hex[ch & 0x0f]);
                } else {
                    result.push_back(static_cast<char>(ch));
                }
        }
    }
    result.push_back('"');
    return result;
}

struct LogitComparison {
    double rmse = 0.0;
    double max_abs = 0.0;
    double cosine = 0.0;
};

LogitComparison compare_logits(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) throw std::runtime_error("logit size mismatch");
    long double squared = 0.0;
    long double dot = 0.0;
    long double norm_a = 0.0;
    long double norm_b = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double difference = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        squared += difference * difference;
        dot += static_cast<long double>(a[i]) * b[i];
        norm_a += static_cast<long double>(a[i]) * a[i];
        norm_b += static_cast<long double>(b[i]) * b[i];
        max_abs = (std::max)(max_abs, std::abs(difference));
    }
    return {
        std::sqrt(static_cast<double>(squared / a.size())),
        max_abs,
        static_cast<double>(dot / std::sqrt(norm_a * norm_b)),
    };
}

void write_token_array(std::ostream & output, const std::vector<llama_token> & tokens) {
    output << '[';
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) output << ',';
        output << tokens[i];
    }
    output << ']';
}
}  // namespace

int main(int argc, char ** argv) {
    llama_model * model = nullptr;
    llama_context * cold_context = nullptr;
    llama_context * warm_context = nullptr;
    try {
        const Options options = parse_options(argc, argv);
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
        if (!options.hybrid_cache_plan.empty()) {
            model_params.hybrid_expert_cache_plan = options.hybrid_cache_plan.c_str();
        }

        const auto process_begin = clock_type::now();
        const auto model_begin = process_begin;
        model = llama_model_load_from_file(options.model.c_str(), model_params);
        const auto model_end = clock_type::now();
        if (model == nullptr) throw std::runtime_error("model load failed");
        const double model_load_ms = elapsed_ms(model_begin, model_end);

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int vocabulary_size = llama_vocab_n_tokens(vocab);
        auto prompt = expand_prompt(tokenize(vocab, apply_chat_template(model, read_file(options.prompt_file))), options.prompt_tokens);

        if (options.cold_only) {
            double context_init_ms = 0.0;
            cold_context = create_context(model, options, context_init_ms);
            const int batch_size = (std::min)(options.n_ctx, options.n_batch);
            const auto prefill_begin = clock_type::now();
            decode_tokens(cold_context, prompt.data(), static_cast<int>(prompt.size()), batch_size);
            const llama_token first_token = sample_greedy(cold_context);
            const std::vector<float> logits = copy_logits(cold_context, vocabulary_size);
            const auto prefill_end = clock_type::now();
            const double prefill_ms = elapsed_ms(prefill_begin, prefill_end);
            const std::vector<llama_token> verification_sequence = generate_tokens(cold_context, options.verify_tokens);
            const std::string verification_text = detokenize(vocab, verification_sequence);

            if (!options.logits_file.empty()) {
                std::ofstream logits_output(options.logits_file, std::ios::binary | std::ios::trunc);
                if (!logits_output) throw std::runtime_error("logits output open failed");
                logits_output.write(reinterpret_cast<const char *>(logits.data()), logits.size() * sizeof(float));
            }

            std::ofstream output(options.output_json, std::ios::binary | std::ios::trunc);
            if (!output) throw std::runtime_error("output JSON open failed");
            output << std::setprecision(12);
            output << "{\n";
            output << "  \"evidence\": \"Measured\",\n";
            output << "  \"mode\": \"cold-only\",\n";
            output << "  \"revision\": \"2d4f3154a2d93c3a4d6d4a415c404f1b397d8dcb\",\n";
            output << "  \"prompt_tokens\": " << prompt.size() << ",\n";
            output << "  \"n_ctx\": " << options.n_ctx << ",\n";
            output << "  \"n_batch\": " << options.n_batch << ",\n";
            output << "  \"n_ubatch\": " << options.n_ubatch << ",\n";
            output << "  \"cache_type_k\": \"" << options.cache_type_k << "\",\n";
            output << "  \"cache_type_v\": \"" << options.cache_type_v << "\",\n";
            output << "  \"model_load_ms\": " << model_load_ms << ",\n";
            output << "  \"context_init_ms\": " << context_init_ms << ",\n";
            output << "  \"cold_prefill_ms\": " << prefill_ms << ",\n";
            output << "  \"cold_prefill_tok_s\": " << prompt.size() * 1000.0 / prefill_ms << ",\n";
            output << "  \"first_token\": " << first_token << ",\n";
            output << "  \"verify_tokens\": ";
            write_token_array(output, verification_sequence);
            output << ",\n";
            output << "  \"verify_text\": " << json_string(verification_text) << "\n";
            output << "}\n";
            std::fprintf(stderr,
                "PREFILL_COLD_ONLY prompt=%zu n_batch=%d n_ubatch=%d prefill_ms=%.3f tok_s=%.3f\n",
                prompt.size(), options.n_batch, options.n_ubatch, prefill_ms,
                prompt.size() * 1000.0 / prefill_ms);

            llama_free(cold_context);
            cold_context = nullptr;
            llama_model_free(model);
            model = nullptr;
            return 0;
        }

        if (options.load_only) {
            double context_init_ms = 0.0;
            warm_context = create_context(model, options, context_init_ms);
            std::vector<llama_token> loaded_tokens(options.n_ctx);
            size_t loaded_token_count = 0;

            const auto load_begin = clock_type::now();
            const bool loaded = llama_state_load_file(
                warm_context, options.state_file.c_str(), loaded_tokens.data(), loaded_tokens.size(), &loaded_token_count);
            const auto load_end = clock_type::now();
            if (!loaded) throw std::runtime_error("state load failed");
            loaded_tokens.resize(loaded_token_count);

            const auto replay_begin = clock_type::now();
            replay_last_token(warm_context, loaded_tokens.back(), static_cast<int32_t>(loaded_tokens.size() - 1));
            const auto replay_end = clock_type::now();
            const auto sample_begin = clock_type::now();
            llama_token token = sample_greedy(warm_context);
            const llama_token first_token = token;
            const auto sample_end = clock_type::now();

            std::vector<double> token_ms;
            token_ms.reserve(options.measured_tokens);
            std::vector<llama_token> generated_tokens(options.measured_tokens + 1);
            generated_tokens[0] = first_token;
            const auto measured_begin = clock_type::now();
            for (int i = 0; i < options.measured_tokens; ++i) {
                const auto begin = clock_type::now();
                if (llama_decode(warm_context, llama_batch_get_one(&token, 1)) != 0) {
                    throw std::runtime_error("measured decode failed");
                }
                token = sample_greedy(warm_context);
                const auto end = clock_type::now();
                token_ms.push_back(elapsed_ms(begin, end));
                generated_tokens[i + 1] = token;
            }
            const auto measured_end = clock_type::now();
            const std::string generated_text = detokenize(vocab, generated_tokens);

            auto ordered = token_ms;
            std::sort(ordered.begin(), ordered.end());
            const double model_load_ms_local = model_load_ms;
            const double load_ms = elapsed_ms(load_begin, load_end);
            const double replay_ms = elapsed_ms(replay_begin, replay_end);
            const double sample_ms = elapsed_ms(sample_begin, sample_end);
            const double warm_ready_ms = load_ms + replay_ms + sample_ms;
            const double measured_ms = elapsed_ms(measured_begin, measured_end);
            const double p50_ms = ordered[ordered.size() / 2];
            const double p95_ms = ordered[(std::min)(ordered.size() - 1, static_cast<size_t>(ordered.size() * 0.95))];
            const bool prompt_tokens_equal = loaded_tokens == prompt;
            const uintmax_t state_file_bytes = std::filesystem::file_size(options.state_file);

            std::ofstream output(options.output_json, std::ios::binary | std::ios::trunc);
            if (!output) throw std::runtime_error("output JSON open failed");
            output << std::setprecision(12);
            output << "{\n";
            output << "  \"evidence\": \"Measured\",\n";
            output << "  \"mode\": \"load-only\",\n";
            output << "  \"revision\": \"2d4f3154a2d93c3a4d6d4a415c404f1b397d8dcb\",\n";
            output << "  \"prompt_tokens\": " << loaded_tokens.size() << ",\n";
            output << "  \"n_ctx\": " << options.n_ctx << ",\n";
            output << "  \"n_batch\": " << options.n_batch << ",\n";
            output << "  \"n_ubatch\": " << options.n_ubatch << ",\n";
            output << "  \"cache_type_k\": \"" << options.cache_type_k << "\",\n";
            output << "  \"cache_type_v\": \"" << options.cache_type_v << "\",\n";
            output << "  \"model_load_ms\": " << model_load_ms_local << ",\n";
            output << "  \"context_init_ms\": " << context_init_ms << ",\n";
            output << "  \"cache_load_ms\": " << load_ms << ",\n";
            output << "  \"last_token_replay_ms\": " << replay_ms << ",\n";
            output << "  \"restored_sampling_ms\": " << sample_ms << ",\n";
            output << "  \"warm_restore_to_first_token_ms\": " << warm_ready_ms << ",\n";
            output << "  \"fresh_process_to_first_token_ms\": "
                   << model_load_ms_local + context_init_ms + warm_ready_ms << ",\n";
            output << "  \"measured_decode_tokens\": " << options.measured_tokens << ",\n";
            output << "  \"restored_decode_ms\": " << measured_ms << ",\n";
            output << "  \"restored_decode_tok_s\": " << options.measured_tokens * 1000.0 / measured_ms << ",\n";
            output << "  \"restored_decode_p50_ms\": " << p50_ms << ",\n";
            output << "  \"restored_decode_p95_ms\": " << p95_ms << ",\n";
            output << "  \"warm_development_iteration_ms\": " << warm_ready_ms + measured_ms << ",\n";
            output << "  \"fresh_process_iteration_ms\": "
                   << model_load_ms_local + context_init_ms + warm_ready_ms + measured_ms << ",\n";
            output << "  \"cache_file_bytes\": " << state_file_bytes << ",\n";
            output << "  \"prompt_tokens_equal\": " << (prompt_tokens_equal ? "true" : "false") << ",\n";
            output << "  \"first_token\": " << first_token << ",\n";
            output << "  \"generated_tokens\": ";
            write_token_array(output, generated_tokens);
            output << ",\n";
            output << "  \"generated_text\": " << json_string(generated_text) << "\n";
            output << "}\n";

            std::fprintf(stderr,
                "PREFIX_CACHE_LOAD_ONLY prompt=%zu load_ms=%.3f replay_ms=%.3f warm_ready_ms=%.3f "
                "decode_tokens=%d decode_tok_s=%.3f iteration_ms=%.3f prompt_equal=%d\n",
                loaded_tokens.size(), load_ms, replay_ms, warm_ready_ms, options.measured_tokens,
                options.measured_tokens * 1000.0 / measured_ms, warm_ready_ms + measured_ms,
                prompt_tokens_equal ? 1 : 0);

            llama_free(warm_context);
            warm_context = nullptr;
            llama_model_free(model);
            model = nullptr;
            return prompt_tokens_equal ? 0 : 2;
        }

        double cold_context_init_ms = 0.0;
        cold_context = create_context(model, options, cold_context_init_ms);
        const int batch_size = (std::min)(options.n_ctx, options.n_batch);

        const auto cold_prefix_begin = clock_type::now();
        decode_tokens(cold_context, prompt.data(), static_cast<int>(prompt.size()) - 1, batch_size);
        const auto cold_prefix_end = clock_type::now();

        const auto save_begin = clock_type::now();
        const bool saved = llama_state_save_file(cold_context, options.state_file.c_str(), prompt.data(), prompt.size());
        const auto save_end = clock_type::now();
        if (!saved) throw std::runtime_error("state save failed");

        const auto cold_last_begin = clock_type::now();
        replay_last_token(cold_context, prompt.back(), static_cast<int32_t>(prompt.size() - 1));
        const auto cold_last_end = clock_type::now();
        const auto cold_sample_begin = clock_type::now();
        const llama_token cold_first_token = sample_greedy(cold_context);
        const auto cold_sample_end = clock_type::now();
        const std::vector<float> cold_logits = copy_logits(cold_context, vocabulary_size);
        if (!options.logits_file.empty()) {
            std::ofstream logits_output(options.logits_file, std::ios::binary | std::ios::trunc);
            if (!logits_output) throw std::runtime_error("logits output open failed");
            logits_output.write(reinterpret_cast<const char *>(cold_logits.data()), cold_logits.size() * sizeof(float));
        }
        const std::vector<llama_token> cold_sequence = generate_tokens(cold_context, options.verify_tokens);
        const std::string cold_verify_text = detokenize(vocab, cold_sequence);

        llama_free(cold_context);
        cold_context = nullptr;

        double warm_context_init_ms = 0.0;
        warm_context = create_context(model, options, warm_context_init_ms);
        std::vector<llama_token> loaded_tokens(options.n_ctx);
        size_t loaded_token_count = 0;
        const auto load_begin = clock_type::now();
        const bool loaded = llama_state_load_file(
            warm_context, options.state_file.c_str(), loaded_tokens.data(), loaded_tokens.size(), &loaded_token_count);
        const auto load_end = clock_type::now();
        if (!loaded) throw std::runtime_error("state load failed");
        loaded_tokens.resize(loaded_token_count);

        const auto replay_begin = clock_type::now();
        replay_last_token(warm_context, loaded_tokens.back(), static_cast<int32_t>(loaded_tokens.size() - 1));
        const auto replay_end = clock_type::now();
        const auto warm_sample_begin = clock_type::now();
        const llama_token warm_first_token = sample_greedy(warm_context);
        const auto warm_sample_end = clock_type::now();
        const std::vector<float> warm_logits = copy_logits(warm_context, vocabulary_size);
        const LogitComparison logit_comparison = compare_logits(cold_logits, warm_logits);
        const std::vector<llama_token> warm_sequence = generate_tokens(warm_context, options.verify_tokens);
        const std::string warm_verify_text = detokenize(vocab, warm_sequence);

        const bool prompt_tokens_equal = loaded_tokens == prompt;
        const bool generated_tokens_equal = warm_sequence == cold_sequence;
        const bool first_token_equal = warm_first_token == cold_first_token;

        llama_token token = sample_greedy(warm_context);
        std::vector<double> token_ms;
        token_ms.reserve(options.measured_tokens);
        const auto measured_begin = clock_type::now();
        for (int i = 0; i < options.measured_tokens; ++i) {
            const auto begin = clock_type::now();
            if (llama_decode(warm_context, llama_batch_get_one(&token, 1)) != 0) {
                throw std::runtime_error("measured decode failed");
            }
            token = sample_greedy(warm_context);
            const auto end = clock_type::now();
            token_ms.push_back(elapsed_ms(begin, end));
        }
        const auto measured_end = clock_type::now();

        const double cold_prefix_ms = elapsed_ms(cold_prefix_begin, cold_prefix_end);
        const double cold_last_ms = elapsed_ms(cold_last_begin, cold_last_end);
        const double cold_sample_ms = elapsed_ms(cold_sample_begin, cold_sample_end);
        const double cold_prefill_ms = cold_prefix_ms + cold_last_ms;
        const double save_ms = elapsed_ms(save_begin, save_end);
        const double load_ms = elapsed_ms(load_begin, load_end);
        const double replay_ms = elapsed_ms(replay_begin, replay_end);
        const double warm_sample_ms = elapsed_ms(warm_sample_begin, warm_sample_end);
        const double warm_ready_ms = load_ms + replay_ms + warm_sample_ms;
        const double process_to_warm_first_ms = model_load_ms + cold_context_init_ms + cold_prefill_ms + save_ms +
            warm_context_init_ms + warm_ready_ms;
        const double measured_ms = elapsed_ms(measured_begin, measured_end);
        const auto ordered = [&]() {
            auto result = token_ms;
            std::sort(result.begin(), result.end());
            return result;
        }();
        const double p50_ms = ordered[ordered.size() / 2];
        const double p95_ms = ordered[(std::min)(ordered.size() - 1, static_cast<size_t>(ordered.size() * 0.95))];
        const uintmax_t state_file_bytes = std::filesystem::file_size(options.state_file);

        std::ofstream output(options.output_json, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("output JSON open failed");
        output << std::setprecision(12);
        output << "{\n";
        output << "  \"evidence\": \"Measured\",\n";
        output << "  \"mode\": \"roundtrip\",\n";
        output << "  \"revision\": \"2d4f3154a2d93c3a4d6d4a415c404f1b397d8dcb\",\n";
        output << "  \"prompt_tokens\": " << prompt.size() << ",\n";
        output << "  \"n_ctx\": " << options.n_ctx << ",\n";
        output << "  \"n_batch\": " << options.n_batch << ",\n";
        output << "  \"n_ubatch\": " << options.n_ubatch << ",\n";
        output << "  \"cache_type_k\": \"" << options.cache_type_k << "\",\n";
        output << "  \"cache_type_v\": \"" << options.cache_type_v << "\",\n";
        output << "  \"model_load_ms\": " << model_load_ms << ",\n";
        output << "  \"cold_context_init_ms\": " << cold_context_init_ms << ",\n";
        output << "  \"cold_prefill_ms_excluding_save\": " << cold_prefill_ms << ",\n";
        output << "  \"cold_prefill_tok_s\": " << prompt.size() * 1000.0 / cold_prefill_ms << ",\n";
        output << "  \"cache_save_ms\": " << save_ms << ",\n";
        output << "  \"cache_file_bytes\": " << state_file_bytes << ",\n";
        output << "  \"warm_context_init_ms\": " << warm_context_init_ms << ",\n";
        output << "  \"cache_load_ms\": " << load_ms << ",\n";
        output << "  \"last_token_replay_ms\": " << replay_ms << ",\n";
        output << "  \"restored_sampling_ms\": " << warm_sample_ms << ",\n";
        output << "  \"warm_restore_to_first_token_ms\": " << warm_ready_ms << ",\n";
        output << "  \"fresh_process_model_plus_context_plus_restore_ms\": "
               << model_load_ms + warm_context_init_ms + warm_ready_ms << ",\n";
        output << "  \"full_roundtrip_process_ms\": " << elapsed_ms(process_begin, measured_end) << ",\n";
        output << "  \"measured_decode_tokens\": " << options.measured_tokens << ",\n";
        output << "  \"restored_decode_ms\": " << measured_ms << ",\n";
        output << "  \"restored_decode_tok_s\": " << options.measured_tokens * 1000.0 / measured_ms << ",\n";
        output << "  \"restored_decode_p50_ms\": " << p50_ms << ",\n";
        output << "  \"restored_decode_p95_ms\": " << p95_ms << ",\n";
        output << "  \"prompt_tokens_equal\": " << (prompt_tokens_equal ? "true" : "false") << ",\n";
        output << "  \"first_token_equal\": " << (first_token_equal ? "true" : "false") << ",\n";
        output << "  \"generated_tokens_equal\": " << (generated_tokens_equal ? "true" : "false") << ",\n";
        output << "  \"logits_rmse\": " << logit_comparison.rmse << ",\n";
        output << "  \"logits_max_abs\": " << logit_comparison.max_abs << ",\n";
        output << "  \"logits_cosine\": " << logit_comparison.cosine << ",\n";
        output << "  \"cold_first_token\": " << cold_first_token << ",\n";
        output << "  \"warm_first_token\": " << warm_first_token << ",\n";
        output << "  \"cold_verify_tokens\": ";
        write_token_array(output, cold_sequence);
        output << ",\n  \"warm_verify_tokens\": ";
        write_token_array(output, warm_sequence);
        output << ",\n  \"cold_verify_text\": " << json_string(cold_verify_text);
        output << ",\n  \"warm_verify_text\": " << json_string(warm_verify_text);
        output << "\n}\n";

        std::fprintf(stderr,
            "PREFIX_CACHE_RESULT prompt=%zu cold_ms=%.3f pp=%.3f save_ms=%.3f load_ms=%.3f "
            "replay_ms=%.3f warm_ready_ms=%.3f cache_bytes=%llu tokens_equal=%d logits_rmse=%.9g "
            "sequence_equal=%d decode_tok_s=%.3f\n",
            prompt.size(), cold_prefill_ms, prompt.size() * 1000.0 / cold_prefill_ms, save_ms, load_ms,
            replay_ms, warm_ready_ms, static_cast<unsigned long long>(state_file_bytes),
            prompt_tokens_equal ? 1 : 0, logit_comparison.rmse, generated_tokens_equal ? 1 : 0,
            options.measured_tokens * 1000.0 / measured_ms);

        llama_free(warm_context);
        llama_model_free(model);
        return prompt_tokens_equal && first_token_equal && generated_tokens_equal ? 0 : 2;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "PREFIX_CACHE_ERROR %s\n", error.what());
        if (cold_context != nullptr) llama_free(cold_context);
        if (warm_context != nullptr) llama_free(warm_context);
        if (model != nullptr) llama_model_free(model);
        return 1;
    }
}
