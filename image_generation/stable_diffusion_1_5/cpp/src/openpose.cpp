#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "cxxopts.hpp"
#include "imwrite.hpp"
#include "lora.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "scheduler_lms_discrete.hpp"

class Timer {
    const decltype(std::chrono::steady_clock::now()) m_start;

public:
    Timer(const std::string& scope) : m_start(std::chrono::steady_clock::now()) {
        (std::cout << scope << ": ").flush();
    }

    ~Timer() {
        auto m_end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration<double, std::milli>(m_end - m_start).count() << " ms" << std::endl;
    }
};

void compile_models(const std::string& model_path, const std::string& device) {

    ov::Core core;
    core.add_extension(TOKENIZERS_LIBRARY_PATH);

    // Text encoder
    {
        Timer t("Loading and compiling text encoder");

    }
}

int main() {
    std::cout << "OpenVINO version: " << ov::get_openvino_version() << std::endl;
    compile_models("/data/models", "CPU");
    return 0;
}