#include "ggml-hrx-catalog.h"

#include "hrx_embedded_catalog.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iterator>
#include <new>
#include <string>
#include <utility>

namespace {

static bool ggml_backend_hrx_read_text_file(const std::string & path, std::string * out_text) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }
    *out_text = std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    return true;
}

static bool ggml_backend_hrx_read_binary_file(const std::string & path, std::vector<uint8_t> * out_data) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }
    *out_data = std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    return true;
}

static std::string ggml_backend_hrx_join_path(const std::string & base, const std::string & leaf) {
    if (base.empty()) {
        return leaf;
    }
    if (base.back() == '/') {
        return base + leaf;
    }
    return base + "/" + leaf;
}

static std::string ggml_backend_hrx_route_index_key(const std::string & target_key, const std::string & op) {
    return target_key + "\n" + op;
}

static std::string ggml_backend_hrx_test_case_index_key(const std::string & target_key, const std::string & family) {
    return target_key + "\n" + family;
}

static bool ggml_backend_hrx_parse_i64(
        const nlohmann::json & value,
        int64_t * out_value) {
    if (value.is_number_integer()) {
        *out_value = value.get<int64_t>();
        return true;
    }
    if (value.is_number_unsigned()) {
        const uint64_t unsigned_value = value.get<uint64_t>();
        if (unsigned_value > static_cast<uint64_t>(INT64_MAX)) {
            return false;
        }
        *out_value = static_cast<int64_t>(unsigned_value);
        return true;
    }
    return false;
}

static void ggml_backend_hrx_parse_shape_domain(
        const nlohmann::json & route,
        ggml_backend_hrx_catalog_route * out_route) {
    const auto shape_domain = route.find("shape_domain");
    if (shape_domain != route.end() && shape_domain->is_object()) {
        for (auto it = shape_domain->begin(); it != shape_domain->end(); ++it) {
            const std::string key = it.key();
            int64_t value = 0;
            if (!ggml_backend_hrx_parse_i64(it.value(), &value)) {
                continue;
            }
            static constexpr const char * k_min_suffix = "_min";
            static constexpr const char * k_max_suffix = "_max";
            if (key.size() > 4 && key.compare(key.size() - 4, 4, k_min_suffix) == 0) {
                out_route->shape_min[key.substr(0, key.size() - 4)] = value;
            } else if (key.size() > 4 && key.compare(key.size() - 4, 4, k_max_suffix) == 0) {
                out_route->shape_max[key.substr(0, key.size() - 4)] = value;
            }
        }
    }

    const auto shape_guards = route.find("shape_guards");
    if (shape_guards != route.end() && shape_guards->is_object()) {
        for (auto it = shape_guards->begin(); it != shape_guards->end(); ++it) {
            const std::string key = it.key();
            int64_t value = 0;
            if (!ggml_backend_hrx_parse_i64(it.value(), &value)) {
                continue;
            }
            static constexpr const char * k_multiple_suffix = "_multiple_of";
            if (key.size() > 12 && key.compare(key.size() - 12, 12, k_multiple_suffix) == 0) {
                out_route->shape_multiple_of[key.substr(0, key.size() - 12)] = value;
            }
        }
    }
}

static void ggml_backend_hrx_parse_supports(
        const nlohmann::json & route,
        ggml_backend_hrx_catalog_route * out_route) {
    const auto supports = route.find("supports");
    if (supports == route.end() || !supports->is_object()) {
        return;
    }
    for (auto it = supports->begin(); it != supports->end(); ++it) {
        if (it.value().is_string()) {
            out_route->supports[it.key()] = it.value().get<std::string>();
        } else if (it.value().is_boolean()) {
            out_route->supports[it.key()] = it.value().get<bool>() ? "true" : "false";
        }
    }
}

static bool ggml_backend_hrx_parse_constraints(
        const nlohmann::json & route,
        ggml_backend_hrx_catalog_route * out_route,
        std::string * out_error) {
    const auto constraints = route.find("constraints");
    if (constraints == route.end()) {
        return true;
    }
    if (!constraints->is_array()) {
        if (out_error) {
            *out_error = "route " + out_route->id + " constraints must be an array";
        }
        return false;
    }
    for (const auto & constraint : *constraints) {
        if (!constraint.is_object() || !constraint.contains("source") || !constraint["source"].is_string()) {
            if (out_error) {
                *out_error = "route " + out_route->id + " has invalid numeric constraint";
            }
            return false;
        }
        ggml_backend_hrx_catalog_constraint parsed;
        parsed.source = constraint["source"].get<std::string>();
        const bool has_eq = constraint.contains("eq");
        const bool has_eq_source = constraint.contains("eq_source");
        if (has_eq == has_eq_source) {
            if (out_error) {
                *out_error = "route " + out_route->id + " constraint must have exactly one of eq or eq_source";
            }
            return false;
        }
        if (has_eq) {
            if (!ggml_backend_hrx_parse_i64(constraint["eq"], &parsed.eq_value)) {
                if (out_error) {
                    *out_error = "route " + out_route->id + " constraint eq must be an integer";
                }
                return false;
            }
            parsed.has_eq_value = true;
        } else if (constraint["eq_source"].is_string()) {
            parsed.eq_source = constraint["eq_source"].get<std::string>();
        } else {
            if (out_error) {
                *out_error = "route " + out_route->id + " constraint eq_source must be a string";
            }
            return false;
        }
        out_route->constraints.push_back(std::move(parsed));
    }
    return true;
}

static void ggml_backend_hrx_parse_abi(
        const nlohmann::json & route,
        ggml_backend_hrx_catalog_route * out_route) {
    const auto abi = route.find("abi");
    if (abi == route.end() || !abi->is_object()) {
        return;
    }
    out_route->binding_count = static_cast<uint32_t>(abi->value("binding_count", 0));
    out_route->parameter_count = static_cast<uint32_t>(abi->value("parameter_count", 0));
    out_route->constant_byte_length = static_cast<uint32_t>(abi->value("constant_byte_length", 0));
}

static bool ggml_backend_hrx_parse_bindings(
        const nlohmann::json & route,
        ggml_backend_hrx_catalog_route * out_route,
        std::string * out_error) {
    const auto specialization = route.find("specialization");
    if (specialization == route.end() || !specialization->is_object()) {
        return true;
    }
    const auto bindings = specialization->find("bindings");
    const auto workload_arguments = specialization->find("workload_arguments");
    if (workload_arguments != specialization->end()) {
        if (!workload_arguments->is_array()) {
            if (out_error) {
                *out_error = "route " + out_route->id + " specialization.workload_arguments must be an array";
            }
            return false;
        }
        for (const auto & source : *workload_arguments) {
            if (!source.is_string()) {
                if (out_error) {
                    *out_error = "route " + out_route->id + " workload arguments must be strings";
                }
                return false;
            }
            out_route->workload_argument_sources.push_back(source.get<std::string>());
        }
    }
    if (bindings == specialization->end()) {
        return true;
    }
    if (!bindings->is_array()) {
        if (out_error) {
            *out_error = "route " + out_route->id + " specialization.bindings must be an array";
        }
        return false;
    }
    for (const auto & binding : *bindings) {
        if (!binding.is_object() || !binding.contains("key") || !binding["key"].is_string()) {
            if (out_error) {
                *out_error = "route " + out_route->id + " has invalid specialization binding";
            }
            return false;
        }
        ggml_backend_hrx_catalog_binding parsed;
        parsed.key = binding["key"].get<std::string>();
        if (binding.contains("value")) {
            if (binding["value"].is_string()) {
                parsed.value = binding["value"].get<std::string>();
            } else if (binding["value"].is_number_integer()) {
                parsed.value = std::to_string(binding["value"].get<int64_t>());
            } else if (binding["value"].is_number_unsigned()) {
                parsed.value = std::to_string(binding["value"].get<uint64_t>());
            } else {
                if (out_error) {
                    *out_error = "route " + out_route->id + " has unsupported binding value";
                }
                return false;
            }
        } else if (binding.contains("source") && binding["source"].is_string()) {
            parsed.source = binding["source"].get<std::string>();
        } else {
            if (out_error) {
                *out_error = "route " + out_route->id + " binding must have value or source";
            }
            return false;
        }
        out_route->bindings.push_back(std::move(parsed));
    }
    return true;
}

static bool ggml_backend_hrx_parse_string_map(
        const nlohmann::json & object,
        const std::string & context,
        std::map<std::string, std::string> * out_map,
        std::string * out_error) {
    if (!object.is_object()) {
        if (out_error) {
            *out_error = context + " must be an object";
        }
        return false;
    }
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (it.value().is_string()) {
            (*out_map)[it.key()] = it.value().get<std::string>();
        } else if (it.value().is_boolean()) {
            (*out_map)[it.key()] = it.value().get<bool>() ? "true" : "false";
        } else {
            if (out_error) {
                *out_error = context + " entries must be strings or booleans";
            }
            return false;
        }
    }
    return true;
}

static bool ggml_backend_hrx_parse_shape_map(
        const nlohmann::json & object,
        const std::string & context,
        std::map<std::string, int64_t> * out_map,
        std::string * out_error) {
    if (!object.is_object()) {
        if (out_error) {
            *out_error = context + " must be an object";
        }
        return false;
    }
    for (auto it = object.begin(); it != object.end(); ++it) {
        int64_t value = 0;
        if (!ggml_backend_hrx_parse_i64(it.value(), &value)) {
            if (out_error) {
                *out_error = context + " entries must be integers";
            }
            return false;
        }
        (*out_map)[it.key()] = value;
    }
    return true;
}

static bool ggml_backend_hrx_parse_test_schedules(
        const nlohmann::json & value,
        const std::string & source,
        ggml_backend_hrx_catalog * catalog,
        std::string * out_error) {
    const auto schedules = value.find("test_schedules");
    if (schedules == value.end()) {
        return true;
    }
    if (!schedules->is_array()) {
        if (out_error) {
            *out_error = source + ": test_schedules must be an array";
        }
        return false;
    }
    for (const auto & schedule : *schedules) {
        if (!schedule.is_object() || schedule.value("schema", "") != "ggml-hrx-test-schedule-v1") {
            if (out_error) {
                *out_error = source + ": invalid test schedule";
            }
            return false;
        }
        const std::string target_key = schedule.value("target_key", "");
        const std::string family = schedule.value("family", "");
        const auto cases = schedule.find("cases");
        if (target_key.empty() || family.empty() || cases == schedule.end() || !cases->is_array()) {
            if (out_error) {
                *out_error = source + ": test schedule missing target_key, family, or cases";
            }
            return false;
        }
        for (const auto & case_json : *cases) {
            if (!case_json.is_object()) {
                if (out_error) {
                    *out_error = source + ": test case entries must be objects";
                }
                return false;
            }
            ggml_backend_hrx_catalog_test_case test_case;
            test_case.target_key = target_key;
            test_case.family = family;
            test_case.id = case_json.value("id", "");
            test_case.op = case_json.value("op", "");
            test_case.expected_route_id = case_json.value("expected_route_id", "");
            test_case.repeat = static_cast<uint32_t>(case_json.value("repeat", 1));
            if (case_json.contains("tolerance") && case_json["tolerance"].is_number()) {
                test_case.tolerance = case_json["tolerance"].get<float>();
            }
            if (test_case.id.empty() || test_case.op.empty() || test_case.expected_route_id.empty() ||
                test_case.repeat == 0) {
                if (out_error) {
                    *out_error = source + ": test case missing id, op, expected_route_id, or repeat";
                }
                return false;
            }
            if (!case_json.contains("supports") ||
                !ggml_backend_hrx_parse_string_map(
                    case_json["supports"], "test case " + test_case.id + " supports", &test_case.supports, out_error)) {
                return false;
            }
            if (!case_json.contains("shape") ||
                !ggml_backend_hrx_parse_shape_map(
                    case_json["shape"], "test case " + test_case.id + " shape", &test_case.shape, out_error)) {
                return false;
            }
            const size_t test_case_index = catalog->test_cases.size();
            catalog->test_cases_by_target_family[
                ggml_backend_hrx_test_case_index_key(test_case.target_key, test_case.family)].push_back(test_case_index);
            catalog->test_cases.push_back(std::move(test_case));
        }
    }
    catalog->test_case_count = catalog->test_cases.size();
    return true;
}

static bool ggml_backend_hrx_parse_catalog_json(
        const std::string & text,
        const std::string & source,
        const std::string & catalog_dir,
        ggml_backend_hrx_catalog * catalog,
        std::string * out_error) {
    nlohmann::json value;
    try {
        value = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception & e) {
        if (out_error) {
            *out_error = "failed to parse " + source + ": " + e.what();
        }
        return false;
    }

    auto require = [&](bool condition, const char * message) {
        if (!condition && out_error && out_error->empty()) {
            *out_error = source + ": " + message;
        }
        return condition;
    };

    if (!require(value.is_object(), "catalog must be an object") ||
        !require(value.value("schema", "") == "ggml-hrx-catalog-v1", "schema must be ggml-hrx-catalog-v1") ||
        !require(value.contains("catalog_id") && value["catalog_id"].is_string(), "catalog_id must be a string") ||
        !require(value.contains("sources") && value["sources"].is_object(), "sources must be an object") ||
        !require(value.contains("artifacts") && value["artifacts"].is_object(), "artifacts must be an object") ||
        !require(value.contains("families") && value["families"].is_array(), "families must be an array") ||
        !require(value.contains("routes") && value["routes"].is_array(), "routes must be an array")) {
        return false;
    }
    if (value.contains("fusions") && !require(value["fusions"].is_array(), "fusions must be an array")) {
        return false;
    }
    if (value.contains("test_schedules") &&
        !require(value["test_schedules"].is_array(), "test_schedules must be an array")) {
        return false;
    }

    catalog->catalog_id = value["catalog_id"].get<std::string>();
    catalog->source = source;
    catalog->source_count = value["sources"].size();
    catalog->family_count = value["families"].size();
    catalog->fusion_count = value.value("fusions", nlohmann::json::array()).size();

    if (catalog_dir.empty()) {
        for (size_t i = 0; i < ggml_hrx_embedded_artifact_count(); ++i) {
            const ggml_hrx_embedded_artifact & embedded = ggml_hrx_embedded_artifacts()[i];
            ggml_backend_hrx_catalog_artifact artifact;
            artifact.id = embedded.id ? embedded.id : "";
            artifact.path = embedded.path ? embedded.path : "";
            artifact.data.assign(embedded.data, embedded.data + embedded.data_size);
            const auto json_artifact = value["artifacts"].find(artifact.id);
            if (json_artifact != value["artifacts"].end() && json_artifact->is_object()) {
                artifact.format = json_artifact->value("format", "");
            }
            catalog->artifact_by_id[artifact.id] = catalog->artifacts.size();
            catalog->artifacts.push_back(std::move(artifact));
        }
    } else {
        for (auto it = value["artifacts"].begin(); it != value["artifacts"].end(); ++it) {
            if (!it.value().is_object() || !it.value().contains("path") || !it.value()["path"].is_string()) {
                if (out_error) {
                    *out_error = source + ": artifact " + it.key() + " must contain path";
                }
                return false;
            }
            ggml_backend_hrx_catalog_artifact artifact;
            artifact.id = it.key();
            artifact.path = it.value()["path"].get<std::string>();
            artifact.format = it.value().value("format", "");
            if (!ggml_backend_hrx_read_binary_file(ggml_backend_hrx_join_path(catalog_dir, artifact.path), &artifact.data)) {
                if (out_error) {
                    *out_error = source + ": failed to read artifact " + artifact.id;
                }
                return false;
            }
            catalog->artifact_by_id[artifact.id] = catalog->artifacts.size();
            catalog->artifacts.push_back(std::move(artifact));
        }
    }
    catalog->artifact_count = catalog->artifacts.size();

    for (const auto & route_json : value["routes"]) {
        if (!route_json.is_object()) {
            if (out_error) {
                *out_error = source + ": route entries must be objects";
            }
            return false;
        }
        ggml_backend_hrx_catalog_route route;
        route.id = route_json.value("id", "");
        route.family = route_json.value("family", "");
        route.op = route_json.value("op", "");
        route.target_key = route_json.value("target_key", "");
        route.source_id = route_json.value("source_id", "");
        route.artifact_id = route_json.value("artifact_id", "");
        route.root_symbol = route_json.value("root_symbol", "");
        route.export_name = route_json.value("export_name", "");
        if (route.id.empty() || route.op.empty() || route.artifact_id.empty() || route.root_symbol.empty()) {
            if (out_error) {
                *out_error = source + ": route missing id, op, artifact_id, or root_symbol";
            }
            return false;
        }
        ggml_backend_hrx_parse_shape_domain(route_json, &route);
        ggml_backend_hrx_parse_supports(route_json, &route);
        if (!ggml_backend_hrx_parse_constraints(route_json, &route, out_error)) {
            return false;
        }
        ggml_backend_hrx_parse_abi(route_json, &route);
        if (!ggml_backend_hrx_parse_bindings(route_json, &route, out_error)) {
            return false;
        }
        const size_t route_index = catalog->routes.size();
        catalog->routes_by_target_op[ggml_backend_hrx_route_index_key(route.target_key, route.op)].push_back(route_index);
        catalog->routes.push_back(std::move(route));
    }
    catalog->route_count = catalog->routes.size();
    if (!ggml_backend_hrx_parse_test_schedules(value, source, catalog, out_error)) {
        return false;
    }
    return true;
}

} // namespace

void ggml_backend_hrx_catalog_deleter::operator()(ggml_backend_hrx_catalog * catalog) const {
    delete catalog;
}

ggml_backend_hrx_catalog_ptr ggml_backend_hrx_load_catalog(
        const char * catalog_dir,
        std::string * out_error) {
    if (out_error) {
        out_error->clear();
    }

    std::string text;
    std::string source = "embedded catalog";
    std::string catalog_dir_string;
    if (catalog_dir && catalog_dir[0]) {
        catalog_dir_string = catalog_dir;
        source = ggml_backend_hrx_join_path(catalog_dir_string, "catalog.json");
        if (!ggml_backend_hrx_read_text_file(source, &text)) {
            if (out_error) {
                *out_error = "failed to read HRX catalog: " + source;
            }
            return nullptr;
        }
    } else {
        text = ggml_hrx_embedded_catalog_json();
    }

    ggml_backend_hrx_catalog_ptr catalog(new (std::nothrow) ggml_backend_hrx_catalog());
    if (!catalog) {
        if (out_error) {
            *out_error = "failed to allocate HRX catalog";
        }
        return nullptr;
    }
    if (!ggml_backend_hrx_parse_catalog_json(text, source, catalog_dir_string, catalog.get(), out_error)) {
        return nullptr;
    }
    return catalog;
}

const ggml_backend_hrx_catalog_artifact * ggml_backend_hrx_catalog_find_artifact(
        const ggml_backend_hrx_catalog & catalog,
        const std::string & artifact_id) {
    const auto it = catalog.artifact_by_id.find(artifact_id);
    if (it == catalog.artifact_by_id.end() || it->second >= catalog.artifacts.size()) {
        return nullptr;
    }
    return &catalog.artifacts[it->second];
}

const ggml_backend_hrx_catalog_route * ggml_backend_hrx_catalog_find_route(
        const ggml_backend_hrx_catalog & catalog,
        const ggml_backend_hrx_catalog_problem & problem) {
    const auto index_it = catalog.routes_by_target_op.find(ggml_backend_hrx_route_index_key(problem.target_key, problem.op));
    if (index_it == catalog.routes_by_target_op.end()) {
        return nullptr;
    }
    for (const size_t route_index : index_it->second) {
        if (route_index >= catalog.routes.size()) {
            continue;
        }
        const auto & route = catalog.routes[route_index];
        bool match = true;
        for (const auto & support : route.supports) {
            const auto it = problem.supports.find(support.first);
            if (it == problem.supports.end() || it->second != support.second) {
                match = false;
                break;
            }
        }
        if (!match) {
            continue;
        }
        for (const auto & guard : route.shape_min) {
            const auto it = problem.shape.find(guard.first);
            if (it == problem.shape.end() || it->second < guard.second) {
                match = false;
                break;
            }
        }
        if (!match) {
            continue;
        }
        for (const auto & guard : route.shape_max) {
            const auto it = problem.shape.find(guard.first);
            if (it == problem.shape.end() || it->second > guard.second) {
                match = false;
                break;
            }
        }
        if (!match) {
            continue;
        }
        for (const auto & guard : route.shape_multiple_of) {
            const auto it = problem.shape.find(guard.first);
            if (it == problem.shape.end() || guard.second <= 0 || it->second % guard.second != 0) {
                match = false;
                break;
            }
        }
        if (!match) {
            continue;
        }
        for (const auto & constraint : route.constraints) {
            const auto source_it = problem.facts.find(constraint.source);
            if (source_it == problem.facts.end()) {
                match = false;
                break;
            }
            int64_t expected = 0;
            if (constraint.has_eq_value) {
                expected = constraint.eq_value;
            } else {
                const auto expected_it = problem.facts.find(constraint.eq_source);
                if (expected_it == problem.facts.end()) {
                    match = false;
                    break;
                }
                expected = expected_it->second;
            }
            if (source_it->second != expected) {
                match = false;
                break;
            }
        }
        if (match) {
            for (const auto & binding : route.bindings) {
                if (!binding.source.empty()) {
                    static constexpr const char * k_shape_prefix = "shape.";
                    std::string shape_key = binding.source;
                    if (shape_key.compare(0, 6, k_shape_prefix) == 0) {
                        shape_key = shape_key.substr(6);
                    }
                    if (problem.shape.find(shape_key) == problem.shape.end()) {
                        match = false;
                        break;
                    }
                }
            }
        }
        if (match) {
            return &route;
        }
    }
    return nullptr;
}

bool ggml_backend_hrx_catalog_make_config_bindings(
        const ggml_backend_hrx_catalog_route & route,
        const ggml_backend_hrx_catalog_problem & problem,
        std::vector<ggml_backend_hrx_catalog_binding> * out_bindings,
        std::string * out_error) {
    if (out_bindings) {
        out_bindings->clear();
    }
    for (const auto & binding : route.bindings) {
        ggml_backend_hrx_catalog_binding resolved;
        resolved.key = binding.key;
        if (!binding.value.empty()) {
            resolved.value = binding.value;
        } else {
            static constexpr const char * k_shape_prefix = "shape.";
            std::string shape_key = binding.source;
            if (shape_key.compare(0, 6, k_shape_prefix) == 0) {
                shape_key = shape_key.substr(6);
            }
            const auto shape_it = problem.shape.find(shape_key);
            if (shape_it == problem.shape.end()) {
                if (out_error) {
                    *out_error = "route " + route.id + " binding " + binding.key +
                        " references missing shape value " + binding.source;
                }
                return false;
            }
            resolved.value = std::to_string(shape_it->second);
        }
        if (out_bindings) {
            out_bindings->push_back(std::move(resolved));
        }
    }
    return true;
}
