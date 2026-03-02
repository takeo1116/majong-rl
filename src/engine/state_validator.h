#pragma once

#include "core/environment_state.h"
#include <string>
#include <vector>

namespace mahjong {
namespace state_validator {

// 検証結果
struct ValidationResult {
    bool valid = true;
    std::vector<std::string> errors;

    // エラーを追加し、valid を false にする
    void add_error(const std::string& msg) {
        valid = false;
        errors.push_back(msg);
    }
};

// EnvironmentState の整合性を検証する
ValidationResult validate(const EnvironmentState& env);

}  // namespace state_validator
}  // namespace mahjong
