#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using json = nlohmann::json;

struct Options {
    std::string model;
    std::string prompt_file;
    std::string markers_manifest;
    std::string output_json;
};

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        auto value = [&](const char * option) {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + option);
            }
            return argv[++i];
        };

        if (std::strcmp(argv[i], "-m") == 0) {
            options.model = value("-m");
        } else if (std::strcmp(argv[i], "--prompt-file") == 0) {
            options.prompt_file = value("--prompt-file");
        } else if (std::strcmp(argv[i], "--markers-manifest") == 0) {
            options.markers_manifest = value("--markers-manifest");
        } else if (std::strcmp(argv[i], "--output-json") == 0) {
            options.output_json = value("--output-json");
        } else {
            throw std::runtime_error(std::string("unknown option: ") + argv[i]);
        }
    }

    if (options.model.empty() || options.prompt_file.empty() || options.markers_manifest.empty()) {
        throw std::runtime_error("required option is missing");
    }
    return options;
}

std::string read_text(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("file open failed: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

json read_json(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("manifest open failed: " + path);
    }
    return json::parse(input);
}

std::string apply_chat_template(const llama_model * model, const std::string & prompt) {
    const llama_chat_message message = {"user", prompt.c_str()};
    const char * chat_template = llama_model_chat_template(model, nullptr);
    const int32_t required = llama_chat_apply_template(chat_template, &message, 1, true, nullptr, 0);
    if (required < 0) {
        throw std::runtime_error("chat template sizing failed");
    }

    std::vector<char> rendered(static_cast<size_t>(required) + 1);
    const int32_t written = llama_chat_apply_template(
        chat_template, &message, 1, true, rendered.data(), static_cast<int32_t>(rendered.size()));
    if (written < 0) {
        throw std::runtime_error("chat template application failed");
    }
    return std::string(rendered.data(), static_cast<size_t>(written));
}

std::vector<llama_token> tokenize(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special) {
    const int32_t text_size = static_cast<int32_t>(text.size());
    int32_t count = llama_tokenize(vocab, text.c_str(), text_size, nullptr, 0, add_special, true);
    if (count >= 0) {
        throw std::runtime_error("token count probe failed");
    }

    std::vector<llama_token> tokens(static_cast<size_t>(-count));
    count = llama_tokenize(
        vocab,
        text.c_str(),
        text_size,
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        true);
    if (count < 0) {
        throw std::runtime_error("tokenization failed");
    }
    tokens.resize(static_cast<size_t>(count));
    return tokens;
}

std::vector<size_t> find_offsets(
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & marker) {
    std::vector<size_t> offsets;
    if (marker.empty() || marker.size() > tokens.size()) {
        return offsets;
    }

    auto position = tokens.begin();
    while (position != tokens.end()) {
        position = std::search(position, tokens.end(), marker.begin(), marker.end());
        if (position == tokens.end()) {
            break;
        }
        offsets.push_back(static_cast<size_t>(std::distance(tokens.begin(), position)));
        ++position;
    }
    return offsets;
}

struct LocatedMarker {
    std::vector<llama_token> tokens;
    size_t offset = 0;
    size_t trimmed_prefix_tokens = 0;
    size_t trimmed_suffix_tokens = 0;
};

LocatedMarker locate_unique_marker(
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<llama_token> & source_tokens,
        const std::string & id) {
    if (source_tokens.empty()) {
        throw std::runtime_error("empty marker token sequence: " + id);
    }
    const size_t max_trim = (std::min)(static_cast<size_t>(16), source_tokens.size() - 1);
    for (size_t total_trim = 0; total_trim <= max_trim; ++total_trim) {
        for (size_t left_trim = 0; left_trim <= total_trim; ++left_trim) {
            const size_t right_trim = total_trim - left_trim;
            if (left_trim + right_trim >= source_tokens.size()) {
                continue;
            }

            std::vector<llama_token> candidate(
                source_tokens.begin() + static_cast<std::ptrdiff_t>(left_trim),
                source_tokens.end() - static_cast<std::ptrdiff_t>(right_trim));
            const std::vector<size_t> offsets = find_offsets(prompt_tokens, candidate);
            if (offsets.size() == 1) {
                return {std::move(candidate), offsets.front(), left_trim, right_trim};
            }
        }
    }
    throw std::runtime_error("no unique stable marker token sequence: " + id);
}

void write_result(const json & result, const std::string & output_path) {
    const std::string text = result.dump(2) + "\n";
    if (output_path.empty()) {
        std::cout << text;
        return;
    }

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("output open failed: " + output_path);
    }
    output << text;
}
}  // namespace

int main(int argc, char ** argv) {
    llama_model * model = nullptr;
    bool backend_initialized = false;
    try {
        const Options options = parse_options(argc, argv);
        const std::string prompt = read_text(options.prompt_file);
        const json manifest = read_json(options.markers_manifest);
        if (!manifest.contains("markers") || !manifest["markers"].is_array() || manifest["markers"].empty()) {
            throw std::runtime_error("manifest markers array is missing or empty");
        }

        llama_backend_init();
        backend_initialized = true;
        llama_model_params model_params = llama_model_default_params();
        model_params.vocab_only = true;
        model = llama_model_load_from_file(options.model.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("model vocabulary load failed");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const std::string rendered = apply_chat_template(model, prompt);
        const std::vector<llama_token> prompt_tokens = tokenize(vocab, rendered, true);

        json marker_results = json::array();
        std::set<std::string> marker_ids;
        for (const json & marker : manifest["markers"]) {
            const std::string id = marker.at("id").get<std::string>();
            const std::string text = marker.at("text").get<std::string>();
            if (id.empty() || text.empty()) {
                throw std::runtime_error("marker id and text must be non-empty");
            }
            if (!marker_ids.insert(id).second) {
                throw std::runtime_error("duplicate marker id: " + id);
            }

            size_t text_occurrences = 0;
            for (size_t position = prompt.find(text); position != std::string::npos; position = prompt.find(text, position + 1)) {
                ++text_occurrences;
            }
            if (text_occurrences != 1) {
                throw std::runtime_error(
                    "marker text occurrence count is " + std::to_string(text_occurrences) + ": " + id);
            }

            const std::vector<llama_token> source_tokens = tokenize(vocab, text, false);
            LocatedMarker located = locate_unique_marker(prompt_tokens, source_tokens, id);

            json item = {
                {"id", id},
                {"text", text},
                {"offset", located.offset},
                {"token_count", located.tokens.size()},
                {"token_ids", located.tokens},
                {"source_token_count", source_tokens.size()},
                {"trimmed_prefix_tokens", located.trimmed_prefix_tokens},
                {"trimmed_suffix_tokens", located.trimmed_suffix_tokens},
                {"occurrences", 1},
                {"text_occurrences", text_occurrences},
            };
            if (marker.contains("target_fraction")) {
                item["target_fraction"] = marker["target_fraction"];
            }
            marker_results.push_back(std::move(item));
        }

        json result = {
            {"schema_version", 1},
            {"prompt_file", options.prompt_file},
            {"markers_manifest", options.markers_manifest},
            {"prompt_bytes", prompt.size()},
            {"post_template_bytes", rendered.size()},
            {"total_tokens", prompt_tokens.size()},
            {"markers", std::move(marker_results)},
        };
        if (manifest.contains("target_context")) {
            result["target_context"] = manifest["target_context"];
        }
        if (manifest.contains("target_actual_prompt_tokens")) {
            result["target_actual_prompt_tokens"] = manifest["target_actual_prompt_tokens"];
        }

        write_result(result, options.output_json);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        if (model != nullptr) {
            llama_model_free(model);
        }
        if (backend_initialized) {
            llama_backend_free();
        }
        std::cerr << "CONTEXT_PROMPT_ERROR " << error.what() << '\n';
        return 1;
    }
}
