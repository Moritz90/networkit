#include "TwoPhaseInfluenceMaximization.h"

#include "../auxiliary/BucketPQ.h"
#include "../auxiliary/Log.h"
#include "../auxiliary/Random.h"
#include "../auxiliary/SignalHandling.h"

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

using Aux::Random::probability;

using NetworKit::count;
using NetworKit::edgeweight;
using NetworKit::Graph;
using NetworKit::index;
using NetworKit::node;

using Model = NetworKit::TwoPhaseInfluenceMaximization::Model;

void randomReverseReachableSet(const Graph& G, Model model, std::function<void(node)> inserter) {
    auto bfsQueue = std::queue<node>{};
    bfsQueue.push(G.randomNode());
    auto visited = std::vector<bool>(G.numberOfNodes(), false);
    if (model == Model::INDEPENDENT_CASCADE) {
        while (!bfsQueue.empty()) {
            node n = bfsQueue.front();
            bfsQueue.pop();
            inserter(n);

            G.forInEdgesOf(n, [&bfsQueue, &visited](node v, edgeweight weight) {
                if (!visited[v] && probability() <= weight) {
                    bfsQueue.push(v);
                    visited[v] = true;
                }
            });
        }
    } else {
        while (!bfsQueue.empty()) {
            node n = bfsQueue.front();
            bfsQueue.pop();
            inserter(n);

            auto randomNumber = probability();
            auto acc = 0.0;
            G.forInEdgesOf(n, [&bfsQueue, &visited, randomNumber, &acc](node v, edgeweight weight) {
                if (!visited[v] && randomNumber > acc && randomNumber <= acc + weight) {
                    bfsQueue.push(v);
                    visited[v] = true;
                }
                acc += weight;
            });
        }
    }
}

using AdjacencyList = std::vector<index>;
using Hyperedge = std::vector<node>;
using HyperEdges = std::vector<Hyperedge>;
using HyperGraphAdjacencyLists = std::vector<AdjacencyList>;

std::unordered_set<node> extractTopK(count k, const HyperEdges& hyperedges,
        const HyperGraphAdjacencyLists& hypergraphAdjacencyLists, std::vector<count>& nodePriorities) {
    auto result = std::unordered_set<node>{};

    assert(hyperedges.size() <= std::numeric_limits<std::int64_t>::max());
    auto priorityQueue = Aux::BucketPQ(hypergraphAdjacencyLists.size(), 0, static_cast<std::int64_t>(hyperedges.size()));
    for (index node = 0; node < hypergraphAdjacencyLists.size(); ++node) {
        priorityQueue.insert(nodePriorities[node], node);
    }

    auto hyperedgesDeleted = std::vector<bool>(hyperedges.size(), false);

    for (count i = 0; i < k; ++i) {
        index node;
        std::tie(std::ignore, node) = priorityQueue.extractMin();
        result.insert(node);
        for (auto edge : hypergraphAdjacencyLists[node]) {
            if (!hyperedgesDeleted[edge]) {
                hyperedgesDeleted[edge] = true;
                for (auto neighbor : hyperedges[edge]) {
                    nodePriorities[neighbor]++;
                    priorityQueue.changeKey(nodePriorities[neighbor], neighbor);
                }
            }
        }
    }

    return result;
}

double refineKpt(const Graph& G, count k, double epsilon, double l, double kpt1) {
    return kpt1;
}

double estimateKpt(const Graph& G, count k, double epsilon, double l, Model model) {
    // TODO: verify that rounding is correct everywhere here
    for (count i = 1; i <= std::ceil(std::log2(G.numberOfNodes() - 1)); ++i) {
        count n = G.numberOfNodes();
        double c = (6 * l * std::log(n) + 6 * std::log(std::log2(n))) * std::exp2(i);
        double sum = 0;
        auto rrSet = std::vector<node>{};
        INFO(std::to_string((count) std::ceil(c)) + " RR sets will be generated to establish a lower bound for the maximum spread");
        for (count j = 0; j < c; ++j) {
            rrSet.clear();
            randomReverseReachableSet(G, model, [&rrSet](node n) { rrSet.push_back(n); });
            count width = 0;
            for (auto n : rrSet) {
                width += G.degreeIn(n);
            }
            sum += 1 - std::pow(1 - static_cast<double>(width) / G.numberOfEdges(), k);
        }
        if (sum / c > 1 / std::exp2(i)) {
            return refineKpt(G, k, epsilon, l, n * sum / (2 * c));
        }
    }
    return refineKpt(G, k, epsilon, l, 1.0);
}

std::unordered_set<node> selectInfluencers(const Graph& G, count k, count theta, Model model) {
    INFO(std::to_string(theta) + " random RR sets will be generated to select the influencers");

    auto hyperedges = HyperEdges(theta);
    auto hypergraphAdjacencyLists = HyperGraphAdjacencyLists(G.numberOfNodes());
    auto hypergraphPriorities = std::vector<count>(G.numberOfNodes(), theta);

    for (index edge = 0; edge < theta; ++edge) {
        randomReverseReachableSet(G, model,
                [&hyperedges, &hypergraphAdjacencyLists, &hypergraphPriorities, edge](node n) {
            hyperedges[edge].push_back(n);
            hypergraphAdjacencyLists[n].push_back(edge);
            hypergraphPriorities[n]--;
        });
    }

    return extractTopK(k, hyperedges, hypergraphAdjacencyLists, hypergraphPriorities);
}

double log_bin(count n, count k) {
    // calculates ln(n choose k), avoiding integer overflow in case n is large. n > k is assumed.
    double result = 0.0;

    for (count i = 0; i < k; ++i) {
        result += std::log(static_cast<double>(n - i) / (k - i));
    }

    return result;
}

}

namespace NetworKit {

TwoPhaseInfluenceMaximization::TwoPhaseInfluenceMaximization(const Graph& G, count k, double epsilon, double l,
        Model model) : Algorithm(), G(G), k(k), epsilon(epsilon), l(l), model(model) {
    if (!G.isWeighted() || !G.isDirected()) {
        throw std::invalid_argument("two-phase influence maximization expects a weighted, directed graph");
    }

    if (k > G.numberOfNodes()) {
        throw std::invalid_argument("the given seed set size is larger than the number of nodes in the given graph");
    }

#ifdef DEBUG
    // check preconditions
    if (model == Model::LINEAR_THRESHOLD) {
        // verify that weights are assigned correctly
        G.parallelForNodes(
            [&G](node n) {
                edgeweight sum = 0.0;
                G.forInEdgesOf(n, [&sum](node u, node v, edgeweight weight) {
                    if (weight > 1.0 || weight < 0.0) {
                        throw std::invalid_argument("edge with invalid weight between " + std::to_string(u) + " and " + std::to_string(v));
                    }
                    sum += weight;
                });
                if (sum > 1.0) {
                    throw std::invalid_argument("sum of weights of incoming edges > 1 at node " + std::to_string(n));
                }
            }
        );
    } else {
        G.parallelForEdges(
            [&G](node u, node v, edgeweight weight) {
                if (weight > 1.0 || weight < 0.0) {
                    throw std::invalid_argument("edge with invalid weight between " + std::to_string(u) + " and " + std::to_string(v));
                }
            }
        );
    }
#endif
}

void TwoPhaseInfluenceMaximization::run() {
    Aux::SignalHandler handler;

    INFO("Phase 1: Parameter estimation");
    double kpt = estimateKpt(G, k, epsilon, l, model);
    handler.assureRunning();

    count n = G.numberOfNodes();
    double lambda = (8 + 2 * epsilon) * n
        * (l * std::log(n) + log_bin(n, k) + std::log(2))
        * std::pow(epsilon, -2);
    count theta = std::ceil(lambda / kpt); // TODO: verify that rounding is correct

    INFO("Phase 2: Node selection");
    influencers = selectInfluencers(G, k, theta, model);

    hasRun = true;
}

}
