#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "htc/ic.hpp"
#include "htc/types.hpp"

namespace {

std::string get_arg(const std::vector<std::string>& args,
                    const std::string& key,
                    const std::string& default_value = "") {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) {
            return args[i + 1];
        }
    }
    return default_value;
}

bool has_arg(const std::vector<std::string>& args, const std::string& key) {
    for (const auto& a : args) {
        if (a == key) {
            return true;
        }
    }
    return false;
}

htc::Count to_count(const std::string& s, const char* name) {
    try {
        return static_cast<htc::Count>(std::stoull(s));
    } catch (...) {
        throw std::invalid_argument(std::string("invalid value for ") + name + ": " + s);
    }
}

double to_double(const std::string& s, const char* name) {
    try {
        return std::stod(s);
    } catch (...) {
        throw std::invalid_argument(std::string("invalid value for ") + name + ": " + s);
    }
}

std::uint64_t to_u64(const std::string& s, const char* name) {
    try {
        return static_cast<std::uint64_t>(std::stoull(s));
    } catch (...) {
        throw std::invalid_argument(std::string("invalid value for ") + name + ": " + s);
    }
}

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  compare_opim11 [options]\n"
        << "Options:\n"
        << "  --opim-bin <path>        default: OPIM1.1/OPIM1.1\n"
        << "  --opim-workdir <path>    default: OPIM1.1\n"
        << "  --graph-file <path>      default: OPIM1.1/graphInfo/facebook\n"
        << "  --graph-name <name>      default: facebook\n"
        << "  --outdir <name>          default: result\n"
        << "  --k <int>                default: 50\n"
        << "  --epsilon <double>       default: 0.1\n"
        << "  --delta <double>         default: 0.1\n"
        << "  --seed <int>             default: 1 (for local OPIMProxy only)\n";
}

htc::ICGraph load_wc_ic_graph(const std::filesystem::path& graph_file) {
    std::ifstream in(graph_file);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open graph file: " + graph_file.string());
    }

    std::uint64_t n64 = 0U;
    std::uint64_t m64 = 0U;
    if (!(in >> n64 >> m64)) {
        throw std::runtime_error("invalid graph header in: " + graph_file.string());
    }
    if (n64 == 0U) {
        throw std::runtime_error("graph node count is zero in: " + graph_file.string());
    }
    if (n64 > static_cast<std::uint64_t>(std::numeric_limits<htc::NodeId>::max())) {
        throw std::runtime_error("graph node count exceeds NodeId range");
    }

    const htc::NodeId n = static_cast<htc::NodeId>(n64);
    std::vector<std::pair<htc::NodeId, htc::NodeId>> edges;
    edges.reserve(static_cast<std::size_t>(m64));
    std::vector<htc::Count> indeg(n, 0U);

    for (std::uint64_t i = 0U; i < m64; ++i) {
        std::uint64_t u64 = 0U;
        std::uint64_t v64 = 0U;
        if (!(in >> u64 >> v64)) {
            throw std::runtime_error("edge line parse failed in: " + graph_file.string());
        }
        if (u64 >= n64 || v64 >= n64) {
            throw std::runtime_error("edge endpoint out of range in: " + graph_file.string());
        }
        const htc::NodeId u = static_cast<htc::NodeId>(u64);
        const htc::NodeId v = static_cast<htc::NodeId>(v64);
        edges.emplace_back(u, v);
        ++indeg[v];
    }

    htc::ICGraph graph(n);
    for (const auto& [u, v] : edges) {
        if (indeg[v] == 0U) {
            throw std::logic_error("indegree unexpectedly zero during WC construction");
        }
        const htc::Prob p = 1.0 / static_cast<double>(indeg[v]);
        graph.add_edge(u, v, p);
    }
    graph.build_indices();
    graph.validate();
    return graph;
}

struct OPIM11Result {
    std::vector<htc::NodeId> seeds;
    htc::Count rr_samples = 0U;
    double approx = 0.0;
    double influence = 0.0;
    double self_influence = 0.0;
    double upper_bound_derived = 0.0;
};

std::string trim_copy(const std::string& s) {
    std::size_t begin = 0U;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }
    std::size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1U]))) {
        --end;
    }
    return s.substr(begin, end - begin);
}

double parse_metric(const std::string& line, const std::string& prefix) {
    if (line.rfind(prefix, 0U) != 0U) {
        throw std::invalid_argument("prefix mismatch when parsing metric");
    }
    const std::string value_str = trim_copy(line.substr(prefix.size()));
    return std::stod(value_str);
}

htc::Count parse_count_metric(const std::string& line, const std::string& prefix) {
    if (line.rfind(prefix, 0U) != 0U) {
        throw std::invalid_argument("prefix mismatch when parsing count metric");
    }
    const std::string value_str = trim_copy(line.substr(prefix.size()));
    return static_cast<htc::Count>(std::stoull(value_str));
}

OPIM11Result parse_opim11_outputs(const std::filesystem::path& result_file,
                                  const std::filesystem::path& seed_file) {
    OPIM11Result result;

    std::ifstream fin(result_file);
    if (!fin.is_open()) {
        throw std::runtime_error("cannot open OPIM1.1 result file: " + result_file.string());
    }
    std::string line;
    while (std::getline(fin, line)) {
        if (line.rfind("Approx.:", 0U) == 0U) {
            result.approx = parse_metric(line, "Approx.:");
        } else if (line.rfind("Influence:", 0U) == 0U) {
            result.influence = parse_metric(line, "Influence:");
        } else if (line.rfind("Self-estimated influence:", 0U) == 0U) {
            result.self_influence = parse_metric(line, "Self-estimated influence:");
        } else if (line.rfind("#RR sets:", 0U) == 0U) {
            result.rr_samples = parse_count_metric(line, "#RR sets:");
        }
    }

    std::ifstream sin(seed_file);
    if (!sin.is_open()) {
        throw std::runtime_error("cannot open OPIM1.1 seed file: " + seed_file.string());
    }
    while (std::getline(sin, line)) {
        const std::string t = trim_copy(line);
        if (t.empty()) {
            continue;
        }
        result.seeds.push_back(static_cast<htc::NodeId>(std::stoul(t)));
    }

    if (result.approx > 0.0) {
        result.upper_bound_derived = result.self_influence / result.approx;
    }

    return result;
}

std::string shell_escape_single_quoted(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8U);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> args;
        args.reserve(static_cast<std::size_t>(argc));
        for (int i = 1; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }

        if (has_arg(args, "--help") || has_arg(args, "-h")) {
            print_usage();
            return EXIT_SUCCESS;
        }

        const std::filesystem::path opim_bin = get_arg(args, "--opim-bin", "OPIM1.1/OPIM1.1");
        const std::filesystem::path opim_workdir = get_arg(args, "--opim-workdir", "OPIM1.1");
        const std::filesystem::path graph_file = get_arg(args, "--graph-file", "OPIM1.1/graphInfo/facebook");
        const std::string graph_name = get_arg(args, "--graph-name", "facebook");
        const std::string outdir = get_arg(args, "--outdir", "result");

        htc::Budget k = 50U;
        if (const std::string v = get_arg(args, "--k"); !v.empty()) {
            k = static_cast<htc::Budget>(to_count(v, "--k"));
        }

        double epsilon = 0.1;
        if (const std::string v = get_arg(args, "--epsilon"); !v.empty()) {
            epsilon = to_double(v, "--epsilon");
        }

        double delta = 0.1;
        if (const std::string v = get_arg(args, "--delta"); !v.empty()) {
            delta = to_double(v, "--delta");
        }

        std::uint64_t seed = 1U;
        if (const std::string v = get_arg(args, "--seed"); !v.empty()) {
            seed = to_u64(v, "--seed");
        }

        const htc::ICGraph ic_graph = load_wc_ic_graph(graph_file);

        htc::OPIMProxy proxy;
        const htc::OPIMProxyResult proxy_result = proxy.run(ic_graph, k, epsilon, delta, seed);

        const std::string out_file = graph_name + "_opim-c_minBound_k" + std::to_string(k) + "_load";
        const std::filesystem::path result_file = opim_workdir / outdir / out_file;
        const std::filesystem::path seed_file = opim_workdir / outdir / "seed" / ("seed_" + out_file);

        const std::string cmd =
            "cd " + shell_escape_single_quoted(opim_workdir.string()) +
            " && " + shell_escape_single_quoted(std::filesystem::absolute(opim_bin).string()) +
            " -func=1" +
            " -dir=graphInfo" +
            " -gname=" + graph_name +
            " -alg=opim-c" +
            " -mode=2" +
            " -seedsize=" + std::to_string(k) +
            " -eps=" + std::to_string(epsilon) +
            " -delta=" + std::to_string(delta) +
            " -model=IC" +
            " -pdist=load" +
            " -outpath=" + outdir;

        const int rc = std::system(cmd.c_str());
        if (rc != 0) {
            throw std::runtime_error("OPIM1.1 execution failed, command exit code=" + std::to_string(rc));
        }

        const OPIM11Result opim11 = parse_opim11_outputs(result_file, seed_file);

        std::unordered_set<htc::NodeId> opim11_set(opim11.seeds.begin(), opim11.seeds.end());
        htc::Count overlap = 0U;
        for (htc::NodeId v : proxy_result.seeds) {
            if (opim11_set.find(v) != opim11_set.end()) {
                ++overlap;
            }
        }

        const double overlap_ratio =
            (k == 0U) ? 0.0 : static_cast<double>(overlap) / static_cast<double>(k);

        std::cout << "graph.num_nodes=" << ic_graph.num_nodes() << "\n";
        std::cout << "graph.num_edges=" << ic_graph.num_edges() << "\n";
        std::cout << "k=" << k << "\n";
        std::cout << "epsilon=" << epsilon << "\n";
        std::cout << "delta=" << delta << "\n";
        std::cout << "proxy.rr_samples=" << proxy_result.rr_samples << "\n";
        std::cout << "opim11.rr_samples=" << opim11.rr_samples << "\n";
        std::cout << "proxy.upper_bound=" << proxy_result.upper_bound << "\n";
        std::cout << "opim11.upper_bound_derived=" << opim11.upper_bound_derived << "\n";
        std::cout << "opim11.approx=" << opim11.approx << "\n";
        std::cout << "opim11.self_influence=" << opim11.self_influence << "\n";
        std::cout << "seed_overlap_count=" << overlap << "\n";
        std::cout << "seed_overlap_ratio=" << overlap_ratio << "\n";

        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "compare_opim11 failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
